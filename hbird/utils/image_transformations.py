# image_transforms.py
# -----------------------------------------------------------------------------
# Drop-in compatible utilities for paired image/target transforms
# (e.g., semantic segmentation). Public API preserved.
# -----------------------------------------------------------------------------

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms

try:
    # Torchvision >= 0.13 preferred
    from torchvision.transforms import InterpolationMode as _IM
    _BILINEAR = _IM.BILINEAR
    _NEAREST = _IM.NEAREST
except Exception:
    # Fallback to PIL constants for older stacks
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST


# ---- Internal helpers --------------------------------------------------------

def _as_tuple_size(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """Normalize size to (H, W). TorchVision accepts int or (H, W)."""
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be int or (h, w), got {size}")
    return int(size[0]), int(size[1])


def _ensure_tensor_target(target: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Convert target to a Tensor if it's not already.

    NOTE: We intentionally mirror your original behavior: PIL targets become
    float tensors in [0, 1] via ToTensor(). This is *not* ideal for class
    labels, but preserves backward compatibility with your code path.
    """
    if isinstance(target, torch.Tensor):
        return target
    return transforms.ToTensor()(target)


def _resize_pair(
    image: Union[Image.Image, torch.Tensor],
    target: torch.Tensor,
    size: Union[int, Sequence[int]],
) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
    """Resize (image, target) with appropriate interpolation."""
    size = _as_tuple_size(size)

    # Antialias is beneficial for bilinear resampling; only applies where supported.
    # We don't expose this as a parameter to preserve the public API.
    try:
        image = F.resize(image, size, interpolation=_BILINEAR, antialias=True)  # type: ignore[arg-type]
    except TypeError:
        image = F.resize(image, size, interpolation=_BILINEAR)  # Older torchvision

    target = F.resize(target, size, interpolation=_NEAREST)  # keep labels crisp
    return image, target


def _resized_crop_pair(
    image: Union[Image.Image, torch.Tensor],
    target: torch.Tensor,
    i: int, j: int, h: int, w: int,
    size: Union[int, Sequence[int]],
) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
    """Apply the same crop+resize to image and target."""
    size = _as_tuple_size(size)
    try:
        image = F.resized_crop(image, i, j, h, w, size, interpolation=_BILINEAR, antialias=True)  # type: ignore[arg-type]
    except TypeError:
        image = F.resized_crop(image, i, j, h, w, size, interpolation=_BILINEAR)

    target = F.resized_crop(target, i, j, h, w, size, interpolation=_NEAREST)
    return image, target


# ---- Public functions (unchanged signatures) --------------------------------

def random_resize_crop(
    image,
    target,
    size: Tuple[int, int] = (256, 256),
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
):
    """
    Randomly crop and resize `image` and `target` **consistently**.

    Parameters
    ----------
    image : PIL.Image or torch.Tensor
        Input image.
    target : PIL.Image or torch.Tensor
        Corresponding target (e.g., segmentation mask). Will be converted to
        a Tensor if not already (preserves your original behavior).
    size : (int, int), default (256, 256)
        Output size (H, W).
    scale : (float, float), default (0.08, 1.0)
        Area scale range for the crop, as in torchvision's RandomResizedCrop.
    ratio : (float, float), default (3/4, 4/3)
        Aspect ratio range for the crop.

    Returns
    -------
    image, target : same types as torchvision F.* returns
        Image is resized with bilinear; target with nearest.
    """
    target = _ensure_tensor_target(target)
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    return _resized_crop_pair(image, target, i, j, h, w, size)


def resize(
    image,
    target,
    size: Tuple[int, int] = (256, 256),
):
    """
    Resize `image` and `target` **consistently**.

    Parameters
    ----------
    image : PIL.Image or torch.Tensor
    target : PIL.Image or torch.Tensor
        Will be converted to Tensor if not already (compatibility with original code).
    size : (int, int), default (256, 256)
        Output size (H, W).

    Returns
    -------
    image, target
    """
    target = _ensure_tensor_target(target)
    return _resize_pair(image, target, size)


def apply_horizontal_flip(image, target):
    """
    Horizontally flip `image` and `target` **consistently**.

    Notes
    -----
    Your original implementation seeded the global RNG to flip both consistently.
    That's unnecessary—`F.hflip` is deterministic—so we remove the global seeding.

    Returns
    -------
    image, target
    """
    target = _ensure_tensor_target(target)
    image = F.hflip(image)
    target = F.hflip(target)
    return image, target


# ---- Public classes (unchanged signatures) ----------------------------------

class RandomResizedCrop(object):
    """
    Callable transform that applies `random_resize_crop` with probability `p`.

    API preserved:
        RandomResizedCrop(size, scale=(0.5, 2), ratio=(3/4, 4/3), probability=1.0)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.5, 2),
        ratio: Tuple[float, float] = (3. / 4., 4. / 3.),
        probability: float = 1.0,
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.probability = float(probability)

    def __call__(self, img, target):
        if random.random() < self.probability:
            return random_resize_crop(img, target, self.size, self.scale, self.ratio)
        return img, target

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, "
                f"ratio={self.ratio}, p={self.probability})")


class RandomHorizontalFlip(object):
    """
    Callable transform that applies a horizontal flip with probability `p`.

    API preserved:
        RandomHorizontalFlip(probability=0.5)
    """

    def __init__(self, probability: float = 0.5):
        self.probability = float(probability)

    def __call__(self, img, target):
        if random.random() < self.probability:
            return apply_horizontal_flip(img, target)
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability})"


class Compose(object):
    """
    Compose paired transforms that operate on (img, target).

    API preserved:
        Compose([t1, t2, ...])

    Each `t` must accept and return `(img, target)`.
    """

    def __init__(self, transforms: Iterable):
        self.transforms: List = list(transforms)

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self) -> str:
        t_str = ",\n  ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([\n  {t_str}\n])"


class Resize(object):
    """
    Callable transform that applies `resize` to (img, target).

    API preserved:
        Resize(size)
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CombTransforms(object):
    """
    Combine three optional transform slots:
      - img_transform:     applied to `img` only
      - tgt_transform:     applied to `tgt` only
      - img_tgt_transform: applied to the pair `(img, tgt)` and must return `(img, tgt)`

    API preserved:
        CombTransforms(img_transform=None, tgt_transform=None, img_tgt_transform=None)

    Example
    -------
    >>> pair = CombTransforms(
    ...     img_transform=some_img_only_aug,
    ...     tgt_transform=some_tgt_only_aug,
    ...     img_tgt_transform=RandomHorizontalFlip(0.5),
    ... )(img, tgt)
    """

    def __init__(self, img_transform=None, tgt_transform=None, img_tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform
        self.img_tgt_transform = img_tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        if self.img_tgt_transform:
            return self.img_tgt_transform(img, tgt)
        return img, tgt

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"img_transform={self.img_transform}, "
                f"tgt_transform={self.tgt_transform}, "
                f"img_tgt_transform={self.img_tgt_transform})")
