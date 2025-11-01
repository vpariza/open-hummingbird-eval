#!/usr/bin/env python3
"""
Enhanced evaluation CLI aligned to the **actual** `hbird_evaluation` API.

Updates in this version
-----------------------
- **Removed fraction sampling flags**. Dataset subsampling is now **entirely controlled by**
  the dataset name (e.g., `voc*0.2` uses 20% of VOC). No per-file-set fractions here.
- Kept support for optional file-set paths (`--train-fs`, `--val-fs`) if your pipeline
  expects explicit lists; they are passed through unchanged to `hbird_evaluation`.

Key features
------------
- Clean, documented flags that map 1:1 to `hbird_evaluation` params.
- Optional TIMM model loading (`--timm-model`), or plug your own model via `build_model()`.
- Simple logging, seeding, CUDA/AMP toggles, and JSON result dump.

`hbird_evaluation` signature (as implemented here)
-------------------------------------------------
```
hbird_evaluation(
    model: Any,
    d_model: int,
    patch_size: int,
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    input_size: int = 224,
    augmentation_epoch: int = 1,
    device: str | Any = "cpu",
    n_neighbours: int = 30,
    nn_method: str = "scann",
    nn_params: Dict[str, Any] | None = None,
    ftr_extr_fn: Any | None = None,
    memory_size: int | None = None,
    num_workers: int = 8,
    ignore_index: int = 255,
    train_fs_path: str | None = None,
    val_fs_path: str | None = None,
) -> (tuple[float, dict[str, Any]] | float)
```

Examples
--------
# Minimal run (ScaNN, 30-NN), dataset-level subsample via name (e.g., 20% VOC of the explicit 1/128 fileset with seed 42)
python eval.py \
  --dataset-name voc*0.2 \
  --data-dir /data/voc \
  --d-model 768 --input-size 448 \
  --batch-size 16 --device cuda --amp \
  --train-fs ./file_sets/voc/1_div_128/trainaug_128_42.txt \
  --val-fs   ./file_sets/voc/val.txt

# Limit memory bank to 10k samples and ask for KNN details
python eval.py \
  --dataset-name ade20k \
  --data-dir /data/ade20k \
  --patch-size 16 --d-model 768 --n-neighbours 40 \
  --memory-size 10000 \
  --nn-method faiss

# Use full dataset, no internal fractioning here
python eval.py \
  --dataset-name cityscapes \
  --data-dir /data/cityscapes \
  --patch-size 16 --d-model 768 

  
# You can also use dinov2 models loaded directly
python eval.py \
  --dataset-name voc \
  --data-dir /data/voc \
  --d-model 768 --input-size 448 \
  --batch-size 16 --device cuda --amp \
  --dinov2 vitb14 \
  --nn-method faiss

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import math
import numpy as np
import torch

# --- Project imports (these should exist in your repo) ---
try:
    from hbird.hbird_eval import hbird_evaluation
except Exception as e:
    print("[eval.py] Failed to import hbird modules. Did you `pip install -e .` at the repo root?"
          f"Original error: {e}", file=sys.stderr)
    raise

# Optional TIMM model loader
try:
    import timm  # type: ignore
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# -----------------------------
# Argument parsing helpers
# -----------------------------

def _positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return iv


def _device_str(value: str) -> str:
    v = value.strip().lower()
    if v in {"cpu", "cuda"} or v.startswith("cuda:"):
        return v
    raise argparse.ArgumentTypeError("device must be 'cpu', 'cuda', or 'cuda:<idx>'")


# -----------------------------
# Dataclass configs
# -----------------------------

@dataclass
class NNBackend:
    nn_method: str = "scann"        # 'scann' | 'faiss'
    n_neighbours: int = 30
    nn_params: Optional[Dict[str, Any]] = None  # e.g. leaves/leaves_to_search for scann


@dataclass
class RunConfig:
    # Core API params
    dataset_name: str
    data_dir: str
    d_model: int
    patch_size: int

    batch_size: int = 64
    input_size: int = 224
    augmentation_epoch: int = 1
    device: str = "cpu"
    memory_size: Optional[int] = None
    num_workers: int = 8
    ignore_index: int = 255

    # Optional file-set paths (full lists)
    train_fs_path: Optional[str] = None
    val_fs_path: Optional[str] = None

    # Model loading convenience
    timm_model: Optional[str] = None  # e.g. 'vit_base_patch16_224'
    dinov2: Optional[str] = None      # one of: vits14, vitb14, vitl14, vitg14
    checkpoint: Optional[str] = None  # path to weights to load


    # Mixed precision & seed
    amp: bool = False
    seed: Optional[int] = 123

    # NN backend
    nn: NNBackend = field(default_factory=NNBackend)

    # Misc
    out: Optional[str] = None
    log_level: str = "INFO"


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)


# -----------------------------
# Model loading
# -----------------------------

def build_model(cfg: RunConfig) -> Any:
    """Create a model compatible with the repository's feature extraction.

    Options (priority order):
    1) --dinov2 {vits14, vitb14, vitl14, vitg14}  (via torch.hub)
    2) --timm-model <name>                        (via timm)
    Otherwise, raise a clear error so users can plug their own model.
    """
    # 1) DINOv2 via torch.hub
    if cfg.dinov2:
        name = cfg.dinov2.lower()
        valid = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
        if name not in valid:
            raise RuntimeError(f"Unsupported --dinov2 '{cfg.dinov2}'. Choose from: {sorted(valid)}")
        try:
            model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 {name} via torch.hub: {e}")
        # Sanity check on d_model
        exp = valid[name]
        if cfg.d_model != exp:
            logging.getLogger("eval").warning(
                "d_model (%d) does not match expected for %s (%d). Proceeding anyway.",
                cfg.d_model, name, exp
            )
        return model

    # 2) TIMM
    if cfg.timm_model:
        if not _HAS_TIMM:
            raise RuntimeError("timm not installed; install it or omit --timm-model and plug your own model")
        model = timm.create_model(cfg.timm_model, pretrained=True)
        if cfg.checkpoint:
            sd = torch.load(cfg.checkpoint, map_location="cpu")
            state_dict = sd.get("state_dict", sd)
            model.load_state_dict(state_dict, strict=False)
        return model

    # 3) User must customize
    raise RuntimeError(
        "No model specified. Provide --dinov2 (vits14/vitb14/vitl14/vitg14), --timm-model, or edit build_model()."
    )

# -----------------------------
# Main runner
# -----------------------------

def run(cfg: RunConfig) -> Dict[str, Any]:
    logger = logging.getLogger("eval")
    logger.info("===== Hummingbird Evaluation =====")
    logger.info("Config: %s", json.dumps(_public_config_dict(cfg), indent=2))

    set_seed(cfg.seed)

    # Resolve device
    device = torch.device(cfg.device if (not cfg.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    if str(device) == "cpu" and cfg.device.startswith("cuda"):
        logger.warning("CUDA requested but not available. Falling back to CPU.")

    # Build/load model
    model = build_model(cfg).to(device)
    model.eval()

    # Default feature-extractor function
    def _default_ftr_extr_fn(m: Any, imgs: torch.Tensor) -> Any:
        """Return patch embeddings only (CLS removed when present), plus a dummy None.

        Special-cases DINOv2 where `forward_features` returns a dict containing
        `x_norm_patchtokens` (already patch tokens without CLS).
        Falls back gracefully across common output formats (Tensor/Dict/Tuple).
        """
        with torch.no_grad():
            out = m.forward_features(imgs) if hasattr(m, "forward_features") else m(imgs)

        # If this looks like DINOv2 output, prefer the normalized patch tokens directly
        if isinstance(out, dict) and isinstance(out.get("x_norm_patchtokens"), torch.Tensor):
            tokens = out["x_norm_patchtokens"]
            if tokens.dim() != 3:
                raise ValueError(f"Expected (B, N, D) for x_norm_patchtokens, got {tuple(tokens.shape)}")
            return tokens, None  # already patch-only, no CLS present

        # Generic token extraction across popular backbones
        def _grab_tokens(o: Any) -> torch.Tensor:
            if isinstance(o, torch.Tensor):
                return o
            if isinstance(o, dict):
                for k in ("x", "last_hidden_state", "tokens", "out", "features"):
                    v = o.get(k)
                    if isinstance(v, torch.Tensor):
                        return v
                for v in o.values():
                    if isinstance(v, torch.Tensor):
                        return v
            if isinstance(o, (list, tuple)):
                for v in o:
                    if isinstance(v, torch.Tensor):
                        return v
            raise TypeError("Could not locate token tensor in model output")

        tokens = _grab_tokens(out)
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B, N, D) token tensor, got shape {tuple(tokens.shape)}")

        # Heuristic: if N-1 is a perfect square, assume CLS present at index 0 and drop it;
        # if N is a perfect square, assume it's already patch-only.
        B, N, D = tokens.shape
        n_root = int(math.isqrt(max(N, 1)))
        if (N - 1) > 0 and math.isqrt(N - 1) ** 2 == (N - 1):
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens
        return patch_tokens, None

    result = hbird_evaluation(
        model=model,
        d_model=cfg.d_model,
        patch_size=cfg.patch_size,
        dataset_name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        input_size=cfg.input_size,
        augmentation_epoch=cfg.augmentation_epoch,
        device=str(device),
        n_neighbours=cfg.nn.n_neighbours,
        nn_method=cfg.nn.nn_method,
        nn_params=cfg.nn.nn_params,
        ftr_extr_fn=_default_ftr_extr_fn,
        memory_size=cfg.memory_size,
        num_workers=cfg.num_workers,
        ignore_index=cfg.ignore_index,
        train_fs_path=cfg.train_fs_path,
        val_fs_path=cfg.val_fs_path,
    )

    # The function may return a float or a (float, dict)
    if isinstance(result, tuple) and len(result) == 2:
        miou, details = result
        summary = {"miou": float(miou), **details}
    else:
        summary = {"miou": float(result)}

    if cfg.out:
        out_dir = os.path.dirname(cfg.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(cfg.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved results to %s", cfg.out)

    logger.info("===== Summary =====")
    for k, v in summary.items():
        try:
            logger.info("%s: %.4f", k, float(v))
        except Exception:
            logger.info("%s: %s", k, v)

    return summary


def _public_config_dict(cfg: RunConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["device"] = str(cfg.device)
    if d.get("nn", {}).get("nn_params"):
        d["nn"]["nn_params"] = {k: d["nn"]["nn_params"][k] for k in sorted(d["nn"]["nn_params"])[:8]}
    return d


# -----------------------------
# CLI builder
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a ViT-style model with the Hummingbird retrieval + soft-label aggregation pipeline,"
            "using the official hbird_evaluation API. Dataset subsampling is controlled by the dataset name (e.g., voc*0.2)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / dataset
    p.add_argument("--dataset-name", required=True,
                   help="Dataset key (supports fractions like 'voc*0.2' to use 20% at the dataset layer).")
    p.add_argument("--data-dir", required=True,
                   help="Root directory for the dataset. Your data module will resolve splits under this root.")

    # Model / representation
    p.add_argument("--d-model", type=_positive_int, required=True,
                   help="Token embedding dimension D (e.g., 768 for ViT-B).")
    p.add_argument("--patch-size", type=_positive_int, required=True,
                   help="Patch size of the backbone (e.g., 16 for ViT-B/16).")

    # Loader & image geometry
    p.add_argument("--batch-size", type=_positive_int, default=64,
                   help="Batch size for evaluation/feature extraction.")
    p.add_argument("--input-size", type=_positive_int, default=224,
                   help="Input image size for resizing/cropping.")
    p.add_argument("--augmentation-epoch", type=_positive_int, default=1,
                   help="Number of augmentation epochs to build memory (as per repo defaults).")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of DataLoader workers.")

    # Device & precision
    p.add_argument("--device", type=_device_str, default="cpu",
                   help="Device string: 'cpu', 'cuda', or 'cuda:<idx>'.")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision (autocast).")

    # NN configuration
    p.add_argument("--n-neighbours", type=_positive_int, default=30,
                   help="Number of neighbors (K) to retrieve per patch.")
    p.add_argument("--nn-method", choices=["scann", "faiss"], default="scann",
                   help="Nearest-neighbor backend.")
    p.add_argument("--nn-param", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra NN param (repeatable), e.g. --nn-param leaves=200 --nn-param leaves_to_search=50")

    # Optional outputs
    p.add_argument("--memory-size", type=int, default=None,
                   help="Cap the memory bank to this many training samples (None = all).")
    p.add_argument("--ignore-index", type=int, default=255,
                   help="Label index to ignore when computing metrics.")

    # File sets (full lists); no fractions here
    p.add_argument("--train-fs", dest="train_fs_path", type=str, default=None,
                   help="Path to file listing TRAIN items (one per line).")
    p.add_argument("--val-fs", dest="val_fs_path", type=str, default=None,
                   help="Path to file listing VAL items (one per line).")

    # Model loading convenience
    p.add_argument("--timm-model", type=str, default=None,
                   help="Name of a TIMM model to instantiate (e.g., vit_base_patch16_224).")
    p.add_argument("--dinov2", type=str, choices=["vits14", "vitb14", "vitl14", "vitg14"], default=None,
                help="Load a DINOv2 backbone via torch.hub (facebookresearch/dinov2).")

    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional checkpoint path to load into the model.")

    # Misc
    p.add_argument("--seed", type=int, default=123, help="Random seed for torch/cuRAND, etc.")
    p.add_argument("--out", type=str, default=None, help="Path to save a JSON summary of results.")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                   help="Console log verbosity.")

    return p


def _parse_nn_params(kv_list: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in kv_list:
        if "=" not in kv:
            raise argparse.ArgumentTypeError(f"Invalid --nn-param '{kv}'. Use KEY=VALUE.")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in {"true", "false"}:
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    nn_params = _parse_nn_params(args.nn_param)

    cfg = RunConfig(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        d_model=args.d_model,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=args.augmentation_epoch,
        device=args.device,
        memory_size=args.memory_size,
        num_workers=args.num_workers,
        ignore_index=args.ignore_index,
        train_fs_path=args.train_fs_path,
        val_fs_path=args.val_fs_path,
        timm_model=args.timm_model,
        dinov2=args.dinov2,
        checkpoint=args.checkpoint,
        amp=bool(args.amp),
        seed=args.seed,
        nn=NNBackend(
            nn_method=args.nn_method,
            n_neighbours=args.n_neighbours,
            nn_params=nn_params or None,
        ),
        out=args.out,
        log_level=args.log_level,
    )

    configure_logging(cfg.log_level)

    try:
        _ = run(cfg)
    except KeyboardInterrupt:
        logging.getLogger("eval").warning("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
