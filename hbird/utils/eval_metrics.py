import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


class PredsmIoU:
    """
    Fast, memory‑efficient mIoU evaluator with optional Hungarian or many‑to‑one matching.

    Key improvements vs. the original:
    - Streaming, O(C_gt*C_pred) memory: maintains a confusion matrix instead of storing all pixels.
    - Vectorized torch ops (GPU-accelerated when available); graceful CPU fallback.
    - Optional distributed sync to aggregate across multiple GPUs.
    - No joblib loops; IoU/precision matrices derived directly from the confusion matrix.
    - Reordered predictions are produced via a fast vectorized remap (kept API-compatible).
    """

    def __init__(
        self,
        num_pred_classes: int,
        num_gt_classes: int,
        device: Optional[torch.device] = None,
        ignore_index: Optional[int] = None,
        prefer_cuda: bool = True,
        store_reordered_preds: bool = True,
    ):
        """
        Args:
            num_pred_classes: number of predicted classes (columns in confusion matrix)
            num_gt_classes:   number of ground-truth classes (rows in confusion matrix)
            device: torch device to run on (defaults to CUDA if available, else CPU)
            ignore_index: optional label to ignore in GT (commonly 255)
            prefer_cuda: prefer CUDA when available
            store_reordered_preds: if True, the object will keep the raw predictions needed
                to output the reordered prediction array in `compute`. Disable if you only
                need scalars (mIoU/TP/FP/FN) to reduce memory.
        """
        self.num_pred_classes = int(num_pred_classes)
        self.num_gt_classes = int(num_gt_classes)
        self.ignore_index = int(ignore_index) if ignore_index is not None else None
        self.store_reordered_preds = bool(store_reordered_preds)

        if device is None:
            if prefer_cuda and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Streaming confusion matrix (rows = GT classes, cols = Pred classes)
        self._conf_mat = torch.zeros(
            (self.num_gt_classes, self.num_pred_classes),
            dtype=torch.int64,
            device=self.device,
        )

        # To reconstruct reordered predictions at the end (optional)
        self._pred_chunks: List[torch.Tensor] = []  # stored as CPU int32 to save VRAM

    @torch.no_grad()
    def reset(self) -> None:
        self._conf_mat.zero_()
        self._pred_chunks.clear()

    @torch.no_grad()
    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """
        Stream a batch.
        Supports arbitrary shapes as long as gt and pred are broadcast‑equivalent; values are class indices.
        """
        if gt.shape != pred.shape:
            raise ValueError(f"Shapes must match. Got gt={gt.shape}, pred={pred.shape}")

        # Flatten to 1D for bincount logic
        gt = gt.to(self.device, non_blocking=True).reshape(-1).long()
        pred = pred.to(self.device, non_blocking=True).reshape(-1).long()

        # Optional ignore index mask
        if self.ignore_index is not None:
            mask = gt.ne(self.ignore_index)
            gt = gt[mask]
            pred = pred[mask]

        # Mask out of range values (robustness)
        valid = (gt >= 0) & (gt < self.num_gt_classes) & (pred >= 0) & (pred < self.num_pred_classes)
        if not torch.all(valid):
            gt = gt[valid]
            pred = pred[valid]

        if gt.numel() == 0:
            # Nothing to accumulate
            return

        # bincount on (gt, pred) pairs -> confusion increments
        idx = gt * self.num_pred_classes + pred
        counts = torch.bincount(idx, minlength=self.num_gt_classes * self.num_pred_classes)
        self._conf_mat += counts.view(self.num_gt_classes, self.num_pred_classes)

        # Optionally retain predictions to produce reordered_preds later
        if self.store_reordered_preds:
            # keep on CPU to avoid VRAM growth
            self._pred_chunks.append(pred.detach().to("cpu", dtype=torch.int32))

    @torch.no_grad()
    def _score_matrix(self, precision_based: bool = False) -> torch.Tensor:
        """Compute (num_gt x num_pred) score matrix using the current confusion matrix.
        If precision_based=False: IoU = TP / (row + col - TP)
        If precision_based=True:  Precision = TP / col
        Returns:
            torch.Tensor [num_gt, num_pred] on self.device (float64 for stability)
        """
        C = self._conf_mat.to(torch.float64)
        row_sum = C.sum(dim=1, keepdim=True)  # (G,1)
        col_sum = C.sum(dim=0, keepdim=True)  # (1,P)
        TP = C  # elementwise
        eps = 1e-8
        if not precision_based:
            denom = row_sum + col_sum - TP
            # Avoid divide-by-zero
            score = TP / torch.clamp(denom, min=eps)
        else:
            # Precision per (gt, pred) uses all predicted for that pred column
            score = TP / torch.clamp(col_sum, min=eps)
        return score

    @torch.no_grad()
    def _many_to_one_mapping(self, precision_based: bool = False) -> torch.Tensor:
        """Return tensor map[pred_class] -> gt_class (len = num_pred_classes).
        Greedy many-to-one: each predicted class assigned to the GT class with max score.
        """
        score = self._score_matrix(precision_based=precision_based)  # (G,P)
        mapping = score.argmax(dim=0)  # (P,) best gt per predicted class
        return mapping.to(torch.int64)

    @torch.no_grad()
    def _hungarian_mapping(self) -> torch.Tensor:
        """Return tensor map[pred_class] -> gt_class using Hungarian assignment on IoU.
        Unmatched predicted classes are mapped to background (0) to mirror original behavior.
        """
        if not _SCIPY_AVAILABLE:
            raise RuntimeError(
                "scipy is not available for Hungarian matching. Install scipy or use many_to_one=True."
            )
        score = self._score_matrix(precision_based=False)  # IoU matrix (G,P)
        iou_cpu = score.detach().to("cpu", dtype=torch.float64).numpy()
        # linear_sum_assignment minimizes cost, so use 1 - IoU
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_cpu)
        mapping = torch.zeros(self.num_pred_classes, dtype=torch.int64)
        mapping[:] = 0  # default to background
        for r, c in zip(row_ind, col_ind):
            mapping[c] = int(r)
        return mapping

    @torch.no_grad()
    def _tp_fp_fn_from_mapping(self, mapping: Optional[torch.Tensor]) -> Tuple[List[int], List[int], List[int]]:
        """Compute TP/FP/FN per GT class given a mapping from pred->gt.
        If mapping is None (linear_probe), we treat predictions as-is (identity),
        i.e., column i contributes to class i only.
        Returns Python lists for compatibility.
        """
        C = self._conf_mat  # (G,P)
        G, P = C.shape
        row_sum = C.sum(dim=1)  # (G,)

        if mapping is None:
            # Identity columns for i < min(G,P); others don't contribute to any class i's FP
            tp: List[int] = []
            fp: List[int] = []
            fn: List[int] = []
            col_sum = C.sum(dim=0)
            for i in range(G):
                if i < P:
                    TP_i = int(C[i, i].item())
                    FP_i = int((col_sum[i] - C[i, i]).item())
                else:
                    TP_i = 0
                    FP_i = 0
                FN_i = int((row_sum[i] - (C[i, i] if i < P else 0)).item())
                tp.append(TP_i)
                fp.append(FP_i)
                fn.append(FN_i)
            return tp, fp, fn

        # Ensure mapping is on same device
        if mapping.device != C.device:
            mapping = mapping.to(C.device)

        # Integer-safe, CUDA-friendly column folding without matmul
        # C_mapped[:, j] = sum_{p: mapping[p] == j} C[:, p]
        C_mapped = torch.zeros((G, G), dtype=C.dtype, device=C.device)
        C_mapped.index_add_(1, mapping, C)

        col_sum_mapped = C_mapped.sum(dim=0)
        tp_t = torch.diag(C_mapped)
        fp_t = col_sum_mapped - tp_t
        fn_t = row_sum - tp_t

        return (
            tp_t.to(torch.int64).tolist(),
            fp_t.to(torch.int64).tolist(),
            fn_t.to(torch.int64).tolist(),
        )

    @torch.no_grad()
    def _miou_from_counts(self, tp: List[int], fp: List[int], fn: List[int]) -> float:
        tp_t = torch.tensor(tp, dtype=torch.float64)
        fp_t = torch.tensor(fp, dtype=torch.float64)
        fn_t = torch.tensor(fn, dtype=torch.float64)
        denom = tp_t + fp_t + fn_t
        iou = tp_t / torch.clamp(denom, min=1e-8)
        return float(iou.mean().item())

    @torch.no_grad()
    def compute(
        self,
        is_global_zero: bool,
        many_to_one: bool = False,
        precision_based: bool = False,
        linear_probe: bool = False,
        sync_distributed: bool = False,
        return_reordered: bool = True,
    ) -> Tuple[float, List[int], List[int], List[int], List[int], float]:
        """
        Compute mIoU and related stats.

        Args:
            is_global_zero: only compute when True (match DDP behavior in caller)
            many_to_one: assign each predicted class to its best GT class (greedy)
            precision_based: use Precision instead of IoU as the matching score
            linear_probe: skip matching; treat predicted class IDs as final labels
            sync_distributed: if True and torch.distributed is initialized, all-reduce
                the confusion matrix across ranks before computing metrics
            return_reordered: include the reordered per‑pixel predictions in the output
                (requires `store_reordered_preds=True` during updates)

        Returns:
            (miou, tp, fp, fn, reordered_preds, matched_bg_fraction)
        """
        if not is_global_zero:
            # Mirror original API: only compute on rank 0 as instructed by caller
            return 0.0, [], [], [], [], 0.0

        # Optional distributed sync to support multi-GPU runs without manual gather
        if sync_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._conf_mat, op=torch.distributed.ReduceOp.SUM)

        # Determine mapping strategy
        if linear_probe:
            mapping = None
            matched_bg_fraction = 0.0
        else:
            if many_to_one:
                mapping = self._many_to_one_mapping(precision_based=precision_based)
                matched_bg_fraction = float((mapping == 0).sum().item() / max(self.num_pred_classes, 1))
            else:
                mapping = self._hungarian_mapping()  # raises if scipy missing
                # Keep original convention
                matched_bg_fraction = 1.0 / max(self.num_gt_classes, 1)

        # Compute counts and mIoU
        tp, fp, fn = self._tp_fp_fn_from_mapping(mapping)
        miou = self._miou_from_counts(tp, fp, fn)

        # Build reordered predictions if requested
        if return_reordered:
            if not self.store_reordered_preds:
                raise RuntimeError(
                    "return_reordered=True requires store_reordered_preds=True during updates."
                )
            pred_all = torch.cat(self._pred_chunks, dim=0)
            if mapping is None:
                reordered = pred_all.to(torch.int64)
            else:
                # vectorized remap
                map_cpu = mapping.detach().to("cpu")
                reordered = map_cpu[pred_all.to(torch.long)]
            reordered_list = reordered.to(torch.int64).tolist()
        else:
            reordered_list = []

        return miou, tp, fp, fn, reordered_list, matched_bg_fraction

    # --- Backwards-compatible adapter API (kept for drop-in use) ---

    @torch.no_grad()
    def compute_miou(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        num_pred: int,
        num_gt: int,
        many_to_one: bool = False,
        precision_based: bool = False,
        linear_probe: bool = False,
    ) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64], List[np.int64], float]:
        """
        Backwards-compatible single-shot API using numpy arrays.
        Builds the confusion matrix once and computes metrics.
        """
        # Reset internal state and stream arrays once for performance
        self.__init__(
            num_pred_classes=num_pred,
            num_gt_classes=num_gt,
            device=self.device,
            ignore_index=self.ignore_index,
            prefer_cuda=(self.device.type == "cuda"),
            store_reordered_preds=True,
        )
        gt_t = torch.from_numpy(pred.astype(np.int64))  # note: we will swap below due to original sig order
        pred_t = torch.from_numpy(gt.astype(np.int64))
        # The original method signature was (gt, pred, num_pred, num_gt) but internal logic expected (pred, gt)
        # This adapter keeps the behavior consistent with the original compute_miou usage.
        # Accumulate
        self.update(pred_t, gt_t)
        # Compute with requested strategy
        miou, tp, fp, fn, reordered, bg = self.compute(
            is_global_zero=True,
            many_to_one=many_to_one,
            precision_based=precision_based,
            linear_probe=linear_probe,
            sync_distributed=False,
            return_reordered=True,
        )
        # Ensure numpy int64 lists for compatibility
        return (
            float(miou),
            [np.int64(x) for x in tp],
            [np.int64(x) for x in fp],
            [np.int64(x) for x in fn],
            [np.int64(x) for x in reordered],
            float(bg),
        )
