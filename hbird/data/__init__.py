
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple, List
import logging

from hbird.utils.io import read_file_set, list_files

from hbird.utils.image_transformations import CombTransforms

# Pascal VOC dataset
from hbird.data.voc.voc_data import VOCDataModule
from hbird.data.voc.voc_tar_data import VOCDataModule as VOCDataModuleTar
# ADE20K dataset
from hbird.data.ade20k.ade20k_data import Ade20kDataModule
from hbird.data.ade20k.ade20k_tar_data import Ade20kDataModule as Ade20kDataModuleTar
# Cityscapes dataset
from hbird.data.cityscapes.cityscapes_data import CityscapesDataModule
from hbird.data.cityscapes.cityscapes_tar_data import CityscapesDataModule as CityscapesDataModuleTar
# COCO dataset
from hbird.data.coco.coco_data import CocoDataModule
from hbird.data.coco.coco_tar_data import CocoDataModule as CocoDataModuleTar

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


def get_dataset(dataset_name, data_dir, batch_size, num_workers, train_transforms, val_transforms, train_fs_path, val_fs_path, **kwargs) -> Any:
    # Optional file sets
    train_file_set = read_file_set(train_fs_path) if train_fs_path is not None else None
    val_file_set = read_file_set(val_fs_path) if val_fs_path is not None else None

    # Optional sampling fraction via "dataset*fract" syntax (unchanged)
    sample_fract: Optional[float] = None
    if "*" in dataset_name:
        parts = dataset_name.split("*")
        dataset_name = parts[0]
        sample_fract = float(parts[1])
        logger.info("Using %.3f fraction of the %s dataset.", sample_fract, dataset_name)

    # Dataset selection (kept identical)
    dataset_size = 0
    num_classes = 0
    ignore_index_local = -1  # overwritten per dataset below

    if dataset_name == "voc":
        # Pascal VOC requires file sets            
        if train_file_set is None:
            train_fs_path=f"{data_dir}/!VOCSegmentation/sets/trainaug.txt" if data_dir.endswith('.tar') else os.path.join(data_dir, "sets", "trainaug.txt")
            train_file_set = read_file_set(train_fs_path)
        if val_file_set is None:
            val_fs_path=f"{data_dir}/!VOCSegmentation/sets/val.txt" if data_dir.endswith('.tar') else os.path.join(data_dir, "sets", "val.txt")
            val_file_set = read_file_set(val_fs_path)

        if sample_fract is not None:
            random.shuffle(train_file_set)
            train_file_set = train_file_set[: int(len(train_file_set) * sample_fract)]
            logger.info("Sampled %d Pascal VOC images for training.", len(train_file_set))

        cls_name = VOCDataModuleTar if '.tar' in data_dir else VOCDataModule
        ignore_index_local = 255
        dataset = cls_name(
            batch_size=batch_size,
            num_workers=num_workers,
            train_split="trainaug",
            val_split="val",
            data_dir=data_dir,
            train_image_transform=train_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            return_masks=True,
            drop_last=False,
            train_file_set=train_file_set,
            val_file_set=val_file_set,
        )
        dataset.setup()

    elif dataset_name == "ade20k":
        if sample_fract is not None:
            if train_file_set is None:
                # Sample from the whole dataset if file set not provided
                search_dir_path = f"{data_dir}/!ade20k" if data_dir.endswith('.tar') else data_dir
                train_file_set = [
                    f.replace(".jpg", "")
                    for f in list_files(os.path.join(search_dir_path, "images", "training"))
                    if f.endswith(".jpg")
                ]
            random.shuffle(train_file_set)
            train_file_set = train_file_set[: int(len(train_file_set) * sample_fract)]
            logger.info("Sampled %d ADE20K images for training.", len(train_file_set))

        ignore_index_local = 0
        cls_name = Ade20kDataModuleTar if '.tar' in data_dir else Ade20kDataModule
        dataset = cls_name(
            data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            train_file_set=train_file_set,
            drop_last=False,
            val_file_set=val_file_set,
        )
        dataset.setup()

    elif dataset_name == "cityscapes":
        if sample_fract is not None:
            if train_file_set is None:
                search_dir_path = f"{data_dir}/!cityscapes/" if data_dir.endswith('.tar') else data_dir
                img_folder = os.path.join(search_dir_path, "leftImg8bit", "train")
                train_file_set = []

                # list_files handles both tar-internal and normal folders
                for filename in list_files(img_folder):
                    if filename.endswith(".png"):
                        # Get the file name without extension or subpath
                        base_name = os.path.basename(filename).split("_leftImg8bit.png")[0]
                        train_file_set.append(base_name)
            random.shuffle(train_file_set)
            train_file_set = train_file_set[: int(len(train_file_set) * sample_fract)]
            logger.info("Sampled %d Cityscapes images for training.", len(train_file_set))

        ignore_index_local = 255
        cls_name = CityscapesDataModuleTar if '.tar' in data_dir else CityscapesDataModule
        dataset = cls_name(
            root=data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False,
            train_file_set=train_file_set,
            val_file_set=val_file_set,
        )
        dataset.setup()

    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index_local = 255

        if sample_fract is not None:
            if train_file_set is None:
                # Sample from the whole dataset if file set not provided
                search_dir_path = os.path.join(data_dir, "images", "train2017")
                train_file_set = list_files(search_dir_path)
            random.shuffle(train_file_set)
            train_file_set = train_file_set[: int(len(train_file_set) * sample_fract)]
            logger.info("Sampled %d COCO images for training.", len(train_file_set))

        cls_name = CocoDataModuleTar if '.tar' in data_dir else CocoDataModule
        dataset = cls_name(
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=data_dir,
            mask_type=mask_type,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            train_file_set=train_file_set,
            drop_last=False,
            val_file_set=val_file_set,
        )
        dataset.setup()

    else:
        raise ValueError("Unknown dataset name")

    # Dataloaders and sizes (unchanged)
    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()

    logger.info(
        "Dataset=%s | train=%d imgs | val loader ready | num_classes=%d | ignore_index=%d",
        dataset_name,
        dataset_size,
        num_classes,
        ignore_index_local,
    )

    return dataset, ignore_index_local
