# hbird/utils/transforms.py
# -----------------------------------------------------------------------------
# Transformation builders for HBird.
# Public API preserved:
#   - IMAGENET_MEAN, IMAGENET_STD
#   - get_hbird_train_transforms_for_imgs(...)
#   - get_hbird_transforms(...)
#   - get_hbird_train_transforms(...)
#   - get_hbird_val_transforms(...)
#   - get_default_train_transforms(...)
#   - get_default_val_transforms(...)
#   - get_default_transforms(...)
# Returns: dicts with keys {"img", "target", "shared"} or 3-tuple of them.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torchvision.transforms as trn
from hbird.utils.image_transformations import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]  
IMAGENET_STD = [0.229, 0.224, 0.255]

# --------------------------------------------------------------------------
# Internal helpers (no public API changes)
# --------------------------------------------------------------------------
def _resize_with_optional_antialias(size: Tuple[int, int]) -> trn.Resize:
    """
    Build a torchvision Resize transform with antialias when supported.
    Keeps behavior stable across torchvision versions.
    """
    try:
        # torchvision >= 0.17
        return trn.Resize(size, antialias=True)
    except TypeError:
        # older versions without `antialias` kwarg
        return trn.Resize(size)


def _build_color_jitter_block(
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    p_brightness: float,
    p_contrast: float,
    p_saturation: float,
    p_hue: float,
) -> List[trn.RandomApply]:
    """
    Construct the color jitter block with independent activation probabilities.
    Using separate RandomApply keeps your original semantics.
    """
    return [
        trn.RandomApply([trn.ColorJitter(brightness=brightness)], p=p_brightness),
        trn.RandomApply([trn.ColorJitter(contrast=contrast)], p=p_contrast),
        trn.RandomApply([trn.ColorJitter(saturation=saturation)], p=p_saturation),
        trn.RandomApply([trn.ColorJitter(hue=hue)], p=p_hue),
    ]


def _build_image_tensor_block(
    mean: Optional[list], std: Optional[list]
) -> trn.Compose:
    """
    Convert to tensor and optionally normalize. If mean/std are None,
    normalization is skipped (matches your logic in validation path).
    """
    steps = [trn.ToTensor()]
    if mean is not None and std is not None:
        steps.append(trn.Normalize(mean=mean, std=std))
    return trn.Compose(steps)


# --------------------------------------------------------------------------
# Public factories (API preserved; docstrings + types added)
# --------------------------------------------------------------------------
def get_hbird_train_transforms_for_imgs(
    input_size: int = 224,
    min_scale_factor: float = 0.5,
    max_scale_factor: float = 2.0,
    brightness_jitter_range: float = 0.1,
    contrast_jitter_range: float = 0.1,
    saturation_jitter_range: float = 0.1,
    hue_jitter_range: float = 0.1,
    brightness_jitter_probability: float = 0.5,
    contrast_jitter_probability: float = 0.5,
    saturation_jitter_probability: float = 0.5,
    hue_jitter_probability: float = 0.5,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
) -> Dict[str, Optional[trn.Compose]]:
    """
    Image-only training transforms (no target / shared). Matches your original
    function but with clearer structure and validation-friendly internals.
    Returns a dict: {"img": <Compose>, "target": None, "shared": None}.
    """
    global_transforms = [
        trn.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(min_scale_factor, max_scale_factor),
        )
    ]
    local_transforms = _build_color_jitter_block(
        brightness=brightness_jitter_range,
        contrast=contrast_jitter_range,
        saturation=saturation_jitter_range,
        hue=hue_jitter_range,
        p_brightness=brightness_jitter_probability,
        p_contrast=contrast_jitter_probability,
        p_saturation=saturation_jitter_probability,
        p_hue=hue_jitter_probability,
    )
    image_train_transform = trn.Compose(
        [*global_transforms, *local_transforms, *_build_image_tensor_block(img_mean, img_std).transforms]
    )
    return {"img": image_train_transform, "target": None, "shared": None}


def get_hbird_transforms(
    input_size: int = 224,
    min_scale_factor: float = 0.5,
    max_scale_factor: float = 2.0,
    brightness_jitter_range: float = 0.1,
    contrast_jitter_range: float = 0.1,
    saturation_jitter_range: float = 0.1,
    hue_jitter_range: float = 0.1,
    brightness_jitter_probability: float = 0.5,
    contrast_jitter_probability: float = 0.5,
    saturation_jitter_probability: float = 0.5,
    hue_jitter_probability: float = 0.5,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
):
    """
    Convenience wrapper that builds train/val/test transforms (dicts) and
    returns a tuple: (train_transforms, val_transforms, test_transforms).
    """
    train_transforms = get_hbird_train_transforms(
        input_size,
        min_scale_factor,
        max_scale_factor,
        brightness_jitter_range,
        contrast_jitter_range,
        saturation_jitter_range,
        hue_jitter_range,
        brightness_jitter_probability,
        contrast_jitter_probability,
        saturation_jitter_probability,
        hue_jitter_probability,
        img_mean,
        img_std,
    )
    val_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    test_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    return train_transforms, val_transforms, test_transforms


def get_hbird_train_transforms(
    input_size: int = 224,
    min_scale_factor: float = 0.5,
    max_scale_factor: float = 2.0,
    brightness_jitter_range: float = 0.1,
    contrast_jitter_range: float = 0.1,
    saturation_jitter_range: float = 0.1,
    hue_jitter_range: float = 0.1,
    brightness_jitter_probability: float = 0.5,
    contrast_jitter_probability: float = 0.5,
    saturation_jitter_probability: float = 0.5,
    hue_jitter_probability: float = 0.5,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
) -> Dict[str, Optional[trn.Compose]]:
    """
    Train-time transforms: image-only per-sample transforms + shared
    paired transforms (apply to image & target consistently).
    Returns: {"img": <Compose>, "target": None, "shared": <Compose>}
    """
    image_train_transform = trn.Compose(
        [
            *_build_color_jitter_block(
                brightness=brightness_jitter_range,
                contrast=contrast_jitter_range,
                saturation=saturation_jitter_range,
                hue=hue_jitter_range,
                p_brightness=brightness_jitter_probability,
                p_contrast=contrast_jitter_probability,
                p_saturation=saturation_jitter_probability,
                p_hue=hue_jitter_probability,
            ),
            *_build_image_tensor_block(img_mean, img_std).transforms,
        ]
    )

    shared_train_transform = Compose(
        [
            RandomResizedCrop(
                size=(input_size, input_size),
                scale=(min_scale_factor, max_scale_factor),
            ),
            # RandomHorizontalFlip(probability=0.1),  # opt-in
        ]
    )

    return {"img": image_train_transform, "target": None, "shared": shared_train_transform}


def get_hbird_val_transforms(
    input_size: int = 224,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
) -> Dict[str, Optional[trn.Compose]]:
    """
    Validation transforms: deterministic resize + tensor(+normalize),
    plus a shared resize for the paired path (img, target).
    """
    image_val_transform = trn.Compose(
        [
            _resize_with_optional_antialias((input_size, input_size)),
            *_build_image_tensor_block(img_mean, img_std).transforms,
        ]
    )

    shared_val_transform = Compose(
        [
            Resize(size=(input_size, input_size)),
        ]
    )
    return {"img": image_val_transform, "target": None, "shared": shared_val_transform}


def get_default_train_transforms(
    input_size: int = 224,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
    min_scale_factor: float = 0.5,
    max_scale_factor: float = 2.0,
) -> Dict[str, Optional[trn.Compose]]:
    """
    A lighter-weight training preset: just ToTensor/Normalize for `img`,
    and RandomResizedCrop in the shared path (paired).
    """
    image_train_transform = _build_image_tensor_block(img_mean, img_std)

    shared_train_transform = Compose(
        [
            RandomResizedCrop(
                size=(input_size, input_size),
                scale=(min_scale_factor, max_scale_factor),
            ),
        ]
    )

    return {"img": image_train_transform, "target": None, "shared": shared_train_transform}


def get_default_val_transforms(
    input_size: int = 224,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
) -> Dict[str, Optional[trn.Compose]]:
    """
    Default validation preset with deterministic resize and optional normalization.
    Mirrors the behavior of your original method, but centralizes antialias handling.
    """
    image_val_transform = trn.Compose(
        [
            _resize_with_optional_antialias((input_size, input_size)),
            *_build_image_tensor_block(img_mean, img_std).transforms,
        ]
    )

    shared_val_transform = Compose([Resize(size=(input_size, input_size))])
    return {"img": image_val_transform, "target": None, "shared": shared_val_transform}


def get_default_transforms(
    input_size: int = 224,
    img_mean: list = IMAGENET_MEAN,
    img_std: list = IMAGENET_STD,
):
    """
    Convenience wrapper returning (train, val, test) dicts using the default presets.
    """
    train = get_default_train_transforms(input_size, img_mean, img_std)
    val = get_default_val_transforms(input_size, img_mean, img_std)
    test = get_default_val_transforms(input_size, img_mean, img_std)
    return train, val, test
