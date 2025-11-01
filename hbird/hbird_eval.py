"""
This module evaluates a patch-level retrieval + soft-label aggregation
setup for semantic segmentation-style datasets. It constructs a memory
of normalized patch features and their (soft) labels from the training
set, then performs nearest-neighbor retrieval for validation queries
and aggregates neighbor labels via cosine-similarity attention.

Author: vpariza
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import logging

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(iterator, *args, **kwargs):
        return iterator

# --- Project-local imports (kept as-is for compatibility) ---
from hbird.models import FeatureExtractor, FeatureExtractorSimple
from hbird.utils.eval_metrics import PredsmIoU

from hbird.utils.transforms import (
    get_hbird_val_transforms,
    get_hbird_train_transforms,
)

from hbird.utils.image_transformations import CombTransforms
from hbird.data import get_dataset

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure a default, non-intrusive handler if the app didn’t configure logging
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s", datefmt="%H:%M:%S"
        # Timestamp only (no date) to keep console tidy; change as needed in your app.
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

class HbirdEvaluation:
    """
    Build a feature+label memory from the training set and evaluate on a validation set
    via K-Nearest Neighbors and cosine-similarity attention aggregation.

    This class preserves original behaviors:
    - Features stored in memory are L2-normalized (no epsilon added to keep results the same).
    - Query features sent to KNN are NOT normalized here (unchanged by design).
    - Label aggregation (cross_attention) performs L2 normalization internally (unchanged).
    - Sampling logic and distributions are unchanged (including uniform perturbation).

    Parameters
    ----------
    feature_extractor : torch.nn.Module
        Module providing `forward_features(x)` -> (features, aux). The features are expected
        to be shaped (batch_size, num_patches, d_model) with `num_patches` derived from
        eval_spatial_resolution^2.
    train_loader : torch.utils.data.DataLoader
        Dataloader over training samples used to build the memory.
    num_classes : int
        Number of semantic classes; used to construct soft labels per patch.
    n_neighbours : int
        Number of neighbors to retrieve during evaluation.
    augmentation_epoch : int
        Number of passes over the training data to augment and accumulate memory.
    device : str or torch.device
        Device for running the feature extractor (e.g., "cuda" or "cpu").
    nn_method : str, {"faiss","scann"}
        Which nearest-neighbor backend to use.
    nn_params : dict, optional
        Extra parameters passed to the NN backend constructors.
    memory_size : int, optional
        If provided, limits the total memory size (number of patch vectors). When set,
        uniform-perturbed least-frequency sampling is used per image to select patches.
    dataset_size : int, optional
        Number of training images. Required to compute per-image sample count when
        `memory_size` is set. (Passed from the high-level function.)
    f_mem_p : str, optional
        Path to save feature memory tensor (torch.save).
    l_mem_p : str, optional
        Path to save label memory tensor (torch.save).
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        train_loader,
        num_classes: int,
        n_neighbours: int=30,
        augmentation_epoch: int=1,
        device: torch.device | str='cpu',
        nn_method: str = "scann",
        nn_params: Optional[Dict[str, Any]] = None,
        memory_size: Optional[int] = None,
        dataset_size: Optional[int] = None,
        f_mem_p: Optional[str] = None,
        l_mem_p: Optional[str] = None,
    ) -> None:
        if nn_params is None:
            nn_params = {}
        self.nn_params = nn_params

        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()

        self.device = device
        self.nn_method = nn_method
        assert self.nn_method in ["faiss", "scann"], "Only faiss and scann are supported"

        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.n_neighbours = n_neighbours
        self.num_classes = num_classes
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p

        self.num_sampled_features: Optional[int] = None

        # The extractor should expose the evaluation spatial resolution
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution

        logger.info(
            "Initializing memory: nn_method=%s, memory_size=%s, augmentation_epoch=%s",
            self.nn_method,
            str(self.memory_size),
            self.augmentation_epoch,
        )

        # When bounded memory is used, pre-allocate (exact match to original logic)
        if self.memory_size is not None:
            if dataset_size is None:
                raise ValueError("dataset_size must be provided when memory_size is set.")
            denom = (dataset_size * self.augmentation_epoch)
            self.num_sampled_features = max(1, self.memory_size // max(1, denom))
            logger.info(
                "Bounded memory: memory_size=%d, dataset_size=%d, augmentation_epoch=%d, "
                "=> num_sampled_features per image=%d",
                self.memory_size,
                dataset_size,
                self.augmentation_epoch,
                self.num_sampled_features,
            )
            self.feature_memory = torch.zeros(
                (self.memory_size, self.feature_extractor.d_model), dtype=torch.float32
            )
            self.label_memory = torch.zeros(
                (self.memory_size, self.num_classes), dtype=torch.float32
            )

        # Build memory from the training set
        filled = self._create_memory(
            train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution
        )

        # Trim partially filled buffers if bounded
        if self.memory_size is not None and filled is not None and filled < self.memory_size:
            logger.info("Trimming pre-allocated memory to filled size: %d -> %d", self.memory_size, filled)
            self.feature_memory = self.feature_memory[:filled].contiguous()
            self.label_memory = self.label_memory[:filled].contiguous()

        # Optionally save to disk
        self._save_memory()

        # For NN backends expecting CPU tensors, keep on CPU (unchanged behavior)
        self.feature_memory = self.feature_memory.cpu()
        self.label_memory = self.label_memory.cpu()

        # Build KNN index
        self._create_nn(self.n_neighbours, nn_method=self.nn_method, **self.nn_params)

    def evaluate(
        self,
        val_loader,
        eval_spatial_resolution: int,
        return_knn_details: bool = False,
        ignore_index: int = 255,
    ):
        """
        Run validation by retrieving neighbors per query patch and aggregating labels via attention.

        Returns
        -------
        jac : torch.Tensor
            Jaccard/IoU score per class (or similar structure) as produced by PredsmIoU.compute().
        details : dict (optional)
            Only when `return_knn_details=True`. Contains concatenated neighbor features/labels and
            cross-attention aggregated labels (for analysis/visualization).
        """
        metric = PredsmIoU(self.num_classes, self.num_classes, ignore_index=ignore_index)
        self.feature_extractor = self.feature_extractor.to(self.device)

        label_hats: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        knns = []
        knns_labels = []
        knns_ca_labels = []

        logger.info("Starting evaluation loop...")
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader, desc="Evaluation loop")):
                x = x.to(self.device)
                _, _, h, w = x.shape
                features, _ = self.feature_extractor.forward_features(x)  # (BS, N, D)
                features = features.cpu()  # Keep on CPU to match memory/NN (unchanged)
                y = (y * 255).long()       # Matches original behavior
                # y[y == 255] = 0  # (Note: original did not remap here during eval; it remapped earlier in memory creation)

                # KNN retrieval (queries are *not* normalized here, on purpose, to keep parity)
                q = features.clone().detach()
                key_features, key_labels = self._find_nearest_key_to_query(q)

                # Cross-attention label aggregation (normalizes q and k internally, unchanged)
                label_hat = self._cross_attention(features, key_features, key_labels)

                if return_knn_details:
                    knns.append(key_features.detach())
                    knns_labels.append(key_labels.detach())
                    knns_ca_labels.append(label_hat.detach())

                # Reshape back to spatial grid and upsample to image size
                bs, n_patches, label_dim = label_hat.shape
                label_hat = (
                    label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim)
                    .permute(0, 3, 1, 2)
                )
                resized_label_hats = F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")

                # Predicted cluster map (argmax)
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)

                label_hats.append(cluster_map.detach())
                all_labels.append(y.detach())

        labels_cat = torch.cat(all_labels)
        label_hats_cat = torch.cat(label_hats)

        # Update metric
        metric.update(labels_cat, label_hats_cat)
        jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)

        if return_knn_details:
            details = {
                "knns": torch.cat(knns),
                "knns_labels": torch.cat(knns_labels),
                "knns_ca_labels": torch.cat(knns_ca_labels),
            }
            logger.info("Evaluation complete (with KNN details).")
            return jac, details

        logger.info("Evaluation complete.")
        return jac

    def _create_nn(self, n_neighbours: int = 30, nn_method: str = "faiss", **kwargs) -> None:
        """Create the nearest-neighbor search index."""
        logger.info("Building NN index: method=%s, k=%d", nn_method, n_neighbours)
        if nn_method == "scann":
            from hbird.nn.search_scann import NearestNeighborSearchScaNN
            self.NN_algorithm = NearestNeighborSearchScaNN(
                self.feature_memory, n_neighbors=n_neighbours, **kwargs
            )
        elif nn_method == "faiss":
            from hbird.nn.search_faiss import NearestNeighborSearchFaiss
            self.NN_algorithm = NearestNeighborSearchFaiss(
                self.feature_memory, n_neighbors=n_neighbours, **kwargs
            )
        else:
            raise ValueError("Unsupported NN method. Choose from {'faiss','scann'}.")

    def _create_memory(self, train_loader, num_classes: int, eval_spatial_resolution: int) -> Optional[int]:
        """
        Populate feature and label memory from the training data.

        If memory_size is None, concatenates all normalized features and labels.
        If memory_size is set, pre-allocated buffers are filled in place.

        Returns
        -------
        filled : int or None
            Number of rows actually written (only meaningful when memory is bounded).
        """
        feature_memory_chunks: List[torch.Tensor] = []
        label_memory_chunks: List[torch.Tensor] = []

        idx = 0  # write cursor for bounded buffers

        logger.info("Creating memory over %d augmentation epoch(s)...", self.augmentation_epoch)

        with torch.no_grad():
            for j in tqdm(range(self.augmentation_epoch), desc="Augmentation loop"):
                for i, (x, y) in enumerate(tqdm(train_loader, desc="Memory Creation loop")):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Original behavior: scale masks to [0..255] and map 255 -> 0
                    y = (y * 255).long()
                    y[y == 255] = 0

                    features, _ = self.feature_extractor.forward_features(x)  # (BS, N, D)
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution

                    # Patchify GT to align with feature patches
                    patchified_gts = self._patchify_gt(y, patch_size)  # (bs, S, S, c*ps*ps)
                    # For each patch, estimate soft label distribution by pixel class frequency
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)  # (bs, S, S, C)

                    if self.memory_size is None:
                        # Unbounded memory: store all normalized features and labels
                        normalized_features = features / torch.norm(features, dim=2, keepdim=True)
                        normalized_features = normalized_features.flatten(0, 1)  # (bs*S*S, D)
                        label = label.flatten(0, 2)                              # (bs*S*S, C)

                        feature_memory_chunks.append(normalized_features.detach().cpu())
                        label_memory_chunks.append(label.detach().cpu())
                    else:
                        # Bounded memory: sample a fixed number of patches per image
                        sampled_features, sampled_indices = self._sample_features(features, patchified_gts, num_classes)

                        # Normalize sampled features (same as unbounded path; no eps to preserve behavior)
                        normalized_sampled_features = sampled_features / torch.norm(
                            sampled_features, dim=2, keepdim=True
                        )  # (bs, K, D)

                        # Prepare labels for gather (bs, S*S, C)
                        label = label.flatten(1, 2)
                        sampled_indices = sampled_indices.to(self.device)  # match device for gather

                        # Gather the labels for sampled patches
                        label_hat = label.gather(
                            1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1])
                        )  # (bs, K, C)

                        # Flatten batch and write into pre-allocated memory
                        normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                        label_hat = label_hat.flatten(0, 1)

                        end = idx + normalized_sampled_features.size(0)
                        self.feature_memory[idx:end] = normalized_sampled_features.detach().cpu()
                        self.label_memory[idx:end] = label_hat.detach().cpu()
                        idx = end

            if self.memory_size is None:
                # Concatenate all chunks
                self.feature_memory = torch.cat(feature_memory_chunks)
                self.label_memory = torch.cat(label_memory_chunks)
                logger.info(
                    "Unbounded memory created: features=%d x %d",
                    self.feature_memory.size(0),
                    self.feature_memory.size(1),
                )
                return self.feature_memory.size(0)
            else:
                logger.info("Bounded memory filled rows: %d", idx)
                return idx

    def _save_memory(self) -> None:
        """Persist memory to disk if file paths are provided (unchanged behavior)."""
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
            logger.info("Saved feature memory to: %s", self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)
            logger.info("Saved label memory to: %s", self.l_mem_p)

    def load_memory(self) -> bool:
        """
        Load previously saved memory (feature and label tensors).

        Returns
        -------
        bool
            True if both files existed and were loaded, False otherwise.
        """
        if (
            self.f_mem_p is not None
            and self.l_mem_p is not None
            and os.path.isfile(self.f_mem_p)
            and os.path.isfile(self.l_mem_p)
        ):
            self.feature_memory = torch.load(self.f_mem_p)
            self.label_memory = torch.load(self.l_mem_p)
            logger.info("Loaded memory from disk.")
            return True
        logger.warning("Memory files not found or paths not provided; skipping load.")
        return False

    # def _sample_features(
    #     self,
    #     features: torch.Tensor,           # (bs, S*S, D)
    #     patchified_gts: torch.Tensor,     # (bs, S, S, P)
    #     num_classes: int,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Sample a fixed number of patches per image based on class-frequency-derived patch scores,
    #     perturbed by uniform noise over non-empty patches (unchanged behavior).

    #     Returns
    #     -------
    #     sampled_features : (bs, K, D)
    #     sampled_indices  : (bs, K)
    #     """
    #     sampled_features: List[torch.Tensor] = []
    #     sampled_indices: List[torch.Tensor] = []

    #     # Iterate per image to preserve exact sampling pattern
    #     # for k, gt in enumerate(tqdm(patchified_gts, desc="Per-image sampling")):
    #     for k, gt in enumerate(patchified_gts):
    #         # Compute per-patch scores and mask of non-empty patches
    #         patch_scores, nonzero_mask = self._get_patch_scores_and_mask(gt, num_classes)

    #         patch_scores = patch_scores.flatten()      # (S*S,)
    #         nonzero_indices = nonzero_mask.flatten()   # (S*S,)

    #         # Keep identical sentinel behavior for empty patches
    #         patch_scores[~nonzero_indices] = 1e6
    #         # Uniform noise for tie-breaking on non-empty patches; keep on CPU to match original
    #         uniform_x = torch.rand(int(nonzero_indices.sum()))
    #         patch_scores[nonzero_indices] *= uniform_x
    #         # Select smallest scores (least frequent classes favored)
    #         _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)
    #         # Index into features (ensure device alignment for indexing)
    #         feat_k = features[k]
    #         samples = feat_k.index_select(0, indices.to(feat_k.device))

    #         sampled_indices.append(indices)
    #         sampled_features.append(samples)

    #     sampled_features_t = torch.stack(sampled_features, dim=0)  # (bs, K, D)
    #     sampled_indices_t = torch.stack(sampled_indices, dim=0)    # (bs, K)
    #     return sampled_features_t, sampled_indices_t

    def _sample_features(
        self,
        features: torch.Tensor,        # (B, S*S, D)
        patchified_gts: torch.Tensor,  # (B, S, S, P)
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized sampling of K patches per image based on class-frequency-derived scores,
        then uniform tie-breaking noise on non-empty patches (CPU RNG preserved).

        Returns
        -------
        sampled_features : (B, K, D)   on features.device
        sampled_indices  : (B, K)      on CPU (like original)
        """
        B, SS, D = features.shape
        S0, S1, P = patchified_gts.shape[1], patchified_gts.shape[2], patchified_gts.shape[3]
        assert SS == S0 * S1, "Mismatch: features' S*S vs patchified_gts spatial grid"

        dev = patchified_gts.device
        K = int(self.num_sampled_features)

        # --- 1) Per-patch class presence for the whole batch (no Python loops) ---
        # Reshape to (B*SS, P) of class ids
        gt_flat = patchified_gts.reshape(B, SS, P).reshape(B * SS, P).to(torch.long)

        # Per-row bincount via offset trick:
        #   counts_per_patch[row, c] = number of pixels == c in that patch
        # Build unique indices: class_id + row_id * num_classes
        row_offsets = torch.arange(B * SS, device=dev, dtype=torch.long).unsqueeze(1) * num_classes
        vals = (gt_flat + row_offsets).reshape(-1)
        counts = torch.bincount(vals, minlength=(B * SS * num_classes)).reshape(B, SS, num_classes)

        # presence[b, p, c] = 1 if class c appears in patch p of image b
        presence = counts > 0  # (B, SS, C) bool

        # --- 2) Per-image class frequency (#patches containing class c) ---
        class_freq = presence.sum(dim=1).to(torch.float32)          # (B, C)
        presence_f = presence.to(torch.float32)                      # (B, SS, C)

        # --- 3) Patch scores = sum over present classes of their image-level frequencies ---
        # Equivalent to: for each patch, dot(presence[b,p,:], class_freq[b,:])
        patch_scores = torch.einsum("bpc,bc->bp", presence_f, class_freq).cpu()  # (B, SS) on CPU
        nonzero_mask = presence.any(dim=2).cpu()                                  # (B, SS) on CPU

        # Keep identical sentinel for empty patches
        patch_scores.masked_fill_(~nonzero_mask, 1e6)

        # --- 4) Uniform tie-breaking noise (preserve exact CPU RNG sequence) ---
        # Original loop drew rand(nonzero_count[b]) per image in order; we emulate exactly.
        nz_counts = nonzero_mask.sum(dim=1).tolist()  # per-image counts in order
        total_nz = sum(nz_counts)
        if total_nz > 0:
            r = torch.rand(total_nz)                  # CPU RNG, same generator as original
            rand_map = torch.ones_like(patch_scores)  # start with 1s; fill only nonzero positions
            start = 0
            for b, cnt in enumerate(nz_counts):
                if cnt:
                    end = start + cnt
                    rand_map[b, nonzero_mask[b]] = r[start:end]
                    start = end
            patch_scores.mul_(rand_map)               # multiply only where nonzero

        # --- 5) Select K smallest scores per image (identical to original semantics) ---
        _, sampled_indices = torch.topk(patch_scores, K, largest=False)  # (B, K) on CPU

        # --- 6) Gather features in one go ---
        idx_dev = sampled_indices.to(features.device)
        sampled_features = features.gather(1, idx_dev.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)

        return sampled_features, sampled_indices

    def _get_patch_scores_and_mask(
        self, gt: torch.Tensor, num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class-frequency-based scores per patch and a mask for non-empty patches.

        Original logic:
          - For each patch, collect the set of present classes (via unique()).
          - class_frequency[c] = number of patches (in the image) containing class c.
          - patch_score[i,j] = sum_c class_frequency[c] over classes present in patch (i,j).
          - A patch is considered "non-empty" if it contains any class id(s).

        This vectorized implementation produces exactly the same values.
        Returns tensors on CPU to preserve later CPU-side RNG and indexing.
        """
        # gt: (S, S, P) with integer class IDs in [0, num_classes-1]
        S0, S1, P = gt.shape

        # Presence: (S, S, C) boolean if class c appears in the patch
        # One-hot can be big; but this is per-image and matches original semantics.
        presence = F.one_hot(gt, num_classes=num_classes).any(dim=2)  # bool on same device as gt

        # class_frequency[c]: number of patches that contain class c
        class_frequency = presence.sum(dim=(0, 1)).to(torch.float32)  # (C,)

        # Scores: for each patch, sum the class frequencies of its present classes
        patch_scores = presence.to(torch.float32).reshape(-1, num_classes) @ class_frequency  # (S*S,)
        patch_scores = patch_scores.view(S0, S1)

        # Non-empty mask: True if any class is present (matches original "len(unique)>0")
        nonzero_mask = presence.any(dim=2)  # (S, S)

        # Move to CPU to keep downstream ops identical to original (CPU RNG for uniform_x, etc.)
        return patch_scores.cpu(), nonzero_mask.cpu()

    @staticmethod
    def _patchify_gt(gt: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Rearrange GT mask into patches aligned with feature patches.

        Parameters
        ----------
        gt : (bs, c, h, w)
        patch_size : int

        Returns
        -------
        (bs, S, S, c*patch_size*patch_size)
            where S = h // patch_size = w // patch_size
        """
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)
        return gt

    def _cross_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: float = 0.02
    ) -> torch.Tensor:
        """
        Soft label aggregation using cosine-similarity attention (unchanged math).

        Args
        ----
        q : (B, N, D)
        k : (B, N, K, D)
        v : (B, N, K, C)
        beta : float
            Temperature scaling factor.

        Returns
        -------
        label_hat : (B, N, C)
        """
        # Normalize q and k (same as original)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        B, N, D = q.shape
        K = k.size(2)

        # (B*N, 1, D) x (B*N, D, K) -> (B*N, 1, K) / beta -> softmax
        q_flat = q.reshape(B * N, 1, D)
        k_flat = k.reshape(B * N, K, D).transpose(1, 2)
        attn = torch.bmm(q_flat, k_flat).squeeze(1) / beta
        attn = F.softmax(attn, dim=-1)  # (B*N, K)

        # Weighted sum of neighbor labels
        v_flat = v.reshape(B * N, K, -1)
        label_hat = torch.bmm(attn.unsqueeze(1), v_flat).squeeze(1)  # (B*N, C)
        return label_hat.view(B, N, -1)

    def _find_nearest_key_to_query(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve nearest neighbor feature and label tensors for query patches.

        Args
        ----
        q : (B, N, D) queries; NOT normalized here (unchanged behavior).

        Returns
        -------
        key_features : (B, N, k, D)
        key_labels   : (B, N, k, C)
        """
        B, N, D = q.shape
        q_flat = q.reshape(B * N, D).contiguous()  # keep original (no normalization here)

        # Nearest-neighbor search (backend returns numpy indices and distances)
        idx_np, _ = self.NN_algorithm.find_nearest_neighbors(q_flat)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device="cpu")  # indices are CPU

        k = self.n_neighbours
        key_features = self.feature_memory.index_select(0, idx.flatten())
        key_labels = self.label_memory.index_select(0, idx.flatten())

        key_features = key_features.view(B, N, k, -1)
        key_labels = key_labels.view(B, N, k, -1)
        return key_features, key_labels


def hbird_evaluation(
    model,
    d_model: int,
    patch_size: int,
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    input_size: int = 224,
    augmentation_epoch: int = 1,
    device: str | torch.device = "cpu",
    return_knn_details: bool = False,
    n_neighbours: int = 30,
    nn_method: str = "scann",
    nn_params: Optional[Dict[str, Any]] = None,
    ftr_extr_fn=None,
    memory_size: Optional[int] = None,
    num_workers: int = 8,
    ignore_index: int = 255,
    train_fs_path: Optional[str] = None,
    val_fs_path: Optional[str] = None,
):
    """
    High-level evaluation entry point (signature unchanged).

    Returns
    -------
    jac : metric output (and optional details dict if return_knn_details=True)
    """
    if nn_params is None:
        nn_params = {}

    eval_spatial_resolution = input_size // patch_size

    # Feature extractor wrapper (unchanged choice)
    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(
            model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model
        )
    else:
        feature_extractor = FeatureExtractorSimple(
            model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model
        )

    # Transforms (unchanged)
    train_transforms_dict = get_hbird_train_transforms(input_size)
    val_transforms_dict = get_hbird_val_transforms(input_size)

    train_transforms = CombTransforms(
        img_transform=train_transforms_dict["img"], tgt_transform=None, img_tgt_transform=train_transforms_dict["shared"]
    )
    val_transforms = CombTransforms(
        img_transform=val_transforms_dict["img"], tgt_transform=None, img_tgt_transform=val_transforms_dict["shared"]
    )

    dataset, ignore_index_local = get_dataset(dataset_name, data_dir, batch_size, num_workers, train_transforms, val_transforms, train_fs_path, val_fs_path)
    # Dataloaders and sizes (unchanged)
    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    evaluator = HbirdEvaluation(
        feature_extractor,
        train_loader,
        num_classes=num_classes,
        n_neighbours=n_neighbours,
        augmentation_epoch=augmentation_epoch,
        device=device,
        nn_method=nn_method,
        nn_params=nn_params,
        memory_size=memory_size,
        dataset_size=dataset_size,
    )

    # Use dataset-specific ignore_index unless the caller overrides with a non-default
    effective_ignore = ignore_index if ignore_index != 255 else ignore_index_local

    return evaluator.evaluate(
        val_loader,
        eval_spatial_resolution=eval_spatial_resolution,
        return_knn_details=return_knn_details,
        ignore_index=effective_ignore,
    )

if __name__ == "__main__":
    # Add project root to path if running this file as a main script (unchanged)
    p = str(pathlib.Path(__file__).parent.resolve()) + "/"
    if p not in sys.path:
        sys.path.append(p)
