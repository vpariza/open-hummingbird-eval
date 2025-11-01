"""
Feature extraction utilities for ViT-like models (DINO, DINOv2, timm, HuggingFace).

This module provides two drop-in classes that are API-compatible with the
original Open Hummingbird evaluation helpers:

- FeatureExtractorSimple: thin wrapper around an arbitrary "feature function"
- FeatureExtractor: robust, auto-detecting extractor that supports multiple
  ViT backbones and returns token-level patch features and (optionally)
  a normalized CLS attention map.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG_LEVEL = os.environ.get("HBIRD_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.WARNING))
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _model_device(module: nn.Module) -> torch.device:
    """Return the device for a module by inspecting its first parameter."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_minmax(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Min-max normalize last dimension of `x` per batch element.

    If the range is (near-)zero, fall back to zeros to avoid NaNs.
    """
    mins, maxs = x.min(dim=-1, keepdim=True).values, x.max(dim=-1, keepdim=True).values
    denom = (maxs - mins).clamp_min(eps)
    out = (x - mins) / denom
    # If denom was ~0 (all values equal), set to 0 (already handled by clamp)
    return out


def _has_attr(obj: object, dotted: str) -> bool:
    """Return True if `obj` has a nested dotted attribute path."""
    cur = obj
    for name in dotted.split("."):
        if not hasattr(cur, name):
            return False
        cur = getattr(cur, name)
    return True


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

class FeatureExtractorSimple(nn.Module):
    """A thin wrapper that delegates to a provided feature function.

    Parameters
    ----------
    vit_model : nn.Module
        The underlying backbone model.
    ftr_extr_fn : Callable[[nn.Module, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]]
        A callable `(model, imgs) -> features or (features, normalized_cls_attention)`.
        If only features are returned, attention is assumed to be unavailable.
    eval_spatial_resolution : int, default=14
        Spatial resolution (S) for visualization/consumers (not used internally).
    d_model : int, default=768
        Embedding dimensionality of the backbone.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        ftr_extr_fn: Callable,
        eval_spatial_resolution: int = 14,
        d_model: int = 768,
    ) -> None:
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.ftr_extr_fn = ftr_extr_fn

    def forward_features(self, imgs: torch.Tensor):
        return self.ftr_extr_fn(self.model, imgs)

    def forward(self, imgs: torch.Tensor):  # convenience
        return self.forward_features(imgs)


# -----------------------------------------------------------------------------
# Auto-detecting FeatureExtractor
# -----------------------------------------------------------------------------

@dataclass
class _Backend:
    name: str
    supports_attn_api: bool = False  # has get_last_selfattention
    supports_intermediate_layers: bool = False  # has get_intermediate_layers
    forward_features_returns_dict: bool = False  # return dict with patch tokens
    patch_key: Optional[str] = None  # e.g., 'x_norm_patchtokens'


class FeatureExtractor(nn.Module):
    """Robust feature extractor for ViT-like models.

    This class auto-detects the backbone family (DINO, DINOv2, timm ViT, HuggingFace ViT/DeiT)
    and exposes a unified API to retrieve per-patch features and, when possible,
    a normalized CLS attention map.

    Parameters
    ----------
    vit_model : nn.Module
        The vision backbone.
    eval_spatial_resolution : int, default=14
        Spatial resolution S for potential downstream visualization.
    d_model : int, default=768
        Embedding dimensionality.
    use_autocast : bool, default=True
        Use autocast (AMP) during feature extraction when on CUDA.
    autocast_dtype : torch.dtype, default=torch.float16
        AMP dtype when autocast is enabled.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        eval_spatial_resolution: int = 14,
        d_model: int = 768,
        use_autocast: bool = True,
        autocast_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

        self._backend = self._select_backend()
        logger.info("[FeatureExtractor] Selected backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor):
        return self.forward_features(imgs)

    def forward_features(
        self, imgs: torch.Tensor, feat: str = "k"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return per-patch features and normalized CLS attention (if available).

        Parameters
        ----------
        imgs : torch.Tensor
            Input batch of shape (B, C, H, W).
        feat : {"k","q","v"}, default="k"
            For DINO-style QKV extraction via hooks; ignored for backbones that
            do not expose qkv (features come from the patch token stream instead).

        Returns
        -------
        features : torch.Tensor
            Patch token features of shape (B, N_patches, D).
        normalized_cls_attention : Optional[torch.Tensor]
            Normalized attention weights from CLS to patches, shape (B, N_patches),
            or None if the backbone does not expose attention.
        """
        device = _model_device(self.model)
        imgs = imgs.to(device, non_blocking=True)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=self.use_autocast and device.type == "cuda", dtype=self.autocast_dtype)
        )

        with torch.inference_mode(), autocast_ctx:
            if self._backend.name == "dino":
                # Use built-in helpers when available
                feats_list = self.model.get_intermediate_layers(imgs)
                feats = feats_list[0][:, 1:, :]  # drop CLS
                attn = self._get_cls_attention_from_api(imgs)
                return feats, attn

            if self._backend.name == "dinov2":
                out = self.model.forward_features(imgs)
                if self._backend.patch_key and isinstance(out, dict):
                    feats = out[self._backend.patch_key]
                else:
                    feats = out  # rare forks may directly return tensor
                # DINOv2 doesn't ship a public get_last_selfattention; keep None
                return feats, None

            if self._backend.name == "timm":
                out = self.model.forward_features(imgs)
                # timm VisionTransformer forward_features returns (B, N+1, D) or a dict
                if isinstance(out, dict):
                    x = out.get("x", None) or out.get("tokens", None) or next(iter(out.values()))
                else:
                    x = out
                feats = x[:, 1:, :]
                return feats, None

            if self._backend.name == "hf":
                # HuggingFace transformers ViT/DeiT
                out = self.model(imgs, output_attentions=True, return_dict=True)
                last_hidden = out.last_hidden_state  # (B, N+1, D)
                feats = last_hidden[:, 1:, :]
                if out.attentions:
                    # Use last layer attentions: (B, heads, N, N)
                    att = out.attentions[-1]
                    cls_to_patches = att[:, :, 0, 1:].mean(dim=1)  # average heads
                    attn = _normalize_minmax(cls_to_patches)
                else:
                    attn = None
                return feats, attn

            # Fallback: generic ViT with blocks[*].attn.qkv — use hook-based QKV
            feats, attn = self.get_intermediate_layer_feats(imgs, feat=feat, layer_num=-1)
            return feats, attn

    def freeze_feature_extractor(self, unfreeze_layers: Optional[Iterable[str]] = None, regex: bool = False) -> None:
        """Freeze all parameters, optionally unfreezing matching layers.

        Parameters
        ----------
        unfreeze_layers : iterable of str, optional
            Substrings (or regex patterns when `regex=True`) to search for in parameter names.
        regex : bool, default=False
            Treat entries in `unfreeze_layers` as regular expressions.
        """
        patterns = list(unfreeze_layers or [])
        for name, p in self.model.named_parameters():
            requires_grad = False
            for pat in patterns:
                if (regex and re.search(pat, name)) or (not regex and pat in name):
                    requires_grad = True
                    break
            p.requires_grad = requires_grad
        logger.info("[FeatureExtractor] Frozen backbone. Unfrozen patterns: %s", patterns)

    def get_intermediate_layer_feats(
        self, imgs: torch.Tensor, feat: str = "k", layer_num: int = -1
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract Q/K/V features of an intermediate layer via a safe forward hook.

        Works with DINO-like implementations where blocks[layer].attn.qkv exists
        and `model.get_last_selfattention` is available for attention extraction.

        Parameters
        ----------
        imgs : (B, C, H, W)
            Input batch.
        feat : {"q", "k", "v"}
            Which tensor to return (excluding CLS token).
        layer_num : int
            Index of the transformer block to hook (default: last layer, -1).
        """
        assert feat in {"q", "k", "v"}
        device = _model_device(self.model)
        imgs = imgs.to(device, non_blocking=True)

        # Locate the qkv module
        if not _has_attr(self.model, f"blocks.{layer_num}.attn.qkv"):
            raise RuntimeError(
                "qkv module not found at model.blocks[%d].attn.qkv; cannot hook QKV. "
                "Use forward_features() instead or ensure a DINO-style backbone. "
                " % layer_num"
            )
        qkv_module: nn.Module = getattr(getattr(self.model.blocks[layer_num], "attn"), "qkv")

        feat_bucket: dict = {}

        def _hook(_module: nn.Module, _input, output):
            feat_bucket["qkv"] = output

        handle = qkv_module.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                # Trigger a forward to populate the hook
                att = self._get_cls_attention_from_api(imgs)
                if att is None and hasattr(self.model, "__call__"):
                    # Fallback forward to fire the hook
                    _ = self.model(imgs)
        finally:
            handle.remove()

        if "qkv" not in feat_bucket:
            raise RuntimeError("QKV hook did not fire; model forward did not traverse qkv module.")

        # Unpack qkv: [B, N, 3, heads, D_heads] -> (q, k, v)
        qkv = feat_bucket["qkv"]
        B, N, three, heads, Dh = qkv.shape  # type: ignore[attr-defined]
        qkv = qkv.reshape(B, N, 3, heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0].transpose(1, 2).reshape(B, N, -1),
            qkv[1].transpose(1, 2).reshape(B, N, -1),
            qkv[2].transpose(1, 2).reshape(B, N, -1),
        )

        mapping = {"q": q, "k": k, "v": v}
        feats = mapping[feat][:, 1:, :]  # drop CLS

        # Attention (if available): average over heads, normalize
        att = self._get_cls_attention_from_api(imgs)
        return feats, att

    # ------------------------------------------------------------------
    # Backend detection & helpers
    # ------------------------------------------------------------------
    def _select_backend(self) -> _Backend:
        m = self.model
        # DINO exposes: get_intermediate_layers, get_last_selfattention
        if all(hasattr(m, x) for x in ("get_intermediate_layers", "get_last_selfattention")):
            return _Backend(name="dino", supports_attn_api=True, supports_intermediate_layers=True)

        # DINOv2 exposes forward_features(dict with 'x_norm_patchtokens')
        if hasattr(m, "forward_features"):
            try:
                # Light probe with meta-like tensor is brittle; better inspect code paths
                # Many DINOv2 forks keep attribute names consistent
                vt = getattr(m, "__class__", type(m)).__name__.lower()
                if "dino" in vt and "v2" in vt:
                    return _Backend(name="dinov2", forward_features_returns_dict=True, patch_key="x_norm_patchtokens")
            except Exception:  # noqa: BLE001 - extremely defensive
                pass

        # timm VisionTransformer has forward_features and often an attr called 'blocks'
        if hasattr(m, "forward_features") and _has_attr(m, "blocks.0.attn"):
            return _Backend(name="timm")

        # HuggingFace transformers ViT/DeiT style models
        if hasattr(m, "config") and hasattr(m, "__call__"):
            conf = getattr(m, "config")
            if hasattr(conf, "model_type") and str(conf.model_type).lower() in {"vit", "deit"}:
                return _Backend(name="hf")

        logger.warning("[FeatureExtractor] Falling back to generic QKV hook backend.")
        return _Backend(name="generic")

    def _get_cls_attention_from_api(self, imgs: torch.Tensor) -> Optional[torch.Tensor]:
        if hasattr(self.model, "get_last_selfattention"):
            att = self.model.get_last_selfattention(imgs)  # (B, heads, N, N)
            cls_to_patches = att[:, :, 0, 1:].mean(dim=1)
            return _normalize_minmax(cls_to_patches)
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Returns the device where the model is stored."""
        return _model_device(self.model)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sanity check with DINO via torch.hub (CPU ok, but slow)
    import torch

    img = torch.randn(1, 3, 224, 224)
    try:
        dino_vit_s16 = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        dino_vit_s16.eval()
        fx = FeatureExtractor(dino_vit_s16)
        feats, attn = fx.forward_features(img)
        print("Feats:", tuple(feats.shape), "Attn:", None if attn is None else tuple(attn.shape))
    except Exception as e:
        logger.warning("DINO torch.hub load failed: %s", e)
