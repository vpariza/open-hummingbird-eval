import os
import io
import tarfile
from pathlib import Path
from typing import Optional, Callable, Tuple, Any, List

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                         'train', 'tvmonitor']

    def __init__(self,
                 data_dir: str,
                 train_split: str,
                 val_split: str,
                 train_image_transform: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 val_image_transform: Optional[Callable] = None,
                 val_target_transform: Optional[Callable] = None,
                 val_transforms: Optional[Callable] = None,
                 shuffle: bool = False,
                 return_masks: bool = False,
                 drop_last: bool = True,
                 train_file_set=None,
                 val_file_set=None):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_transforms = val_transforms
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks
        self.val_file_set = val_file_set
        self.train_file_set = train_file_set

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        assert train_split in ("trainaug", "train")
        self.voc_train = VOCDataset(root=self.root,
                                    image_set=train_split,
                                    transforms=self.train_image_transform,
                                    file_set=self.train_file_set,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root,
                                  image_set=val_split,
                                  transform=self.val_image_transform,
                                  target_transform=self.val_target_transform,
                                  transforms=self.val_transforms,
                                  file_set=self.val_file_set)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.voc_train)

    def get_val_dataset_size(self):
        return len(self.voc_val)

    def get_num_classes(self):
        return len(self.CLASS_IDX_TO_NAME)


class TrainXVOCValDataModule(pl.LightningDataModule):
    # wrapper class to allow for training on a different data set

    def __init__(self, train_datamodule: pl.LightningDataModule, val_datamodule: VOCDataModule):
        super().__init__()
        self.train_datamodule = train_datamodule
        self.val_datamodule = val_datamodule

    def setup(self, stage: str = None):
        self.train_datamodule.setup(stage)
        self.val_datamodule.setup(stage)

    def class_id_to_name(self, i: int):
        return self.val_datamodule.class_id_to_name(i)

    def __len__(self):
        return len(self.train_datamodule)

    def train_dataloader(self):
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self):
        return self.val_datamodule.val_dataloader()


class VOCDataset(VisionDataset):

    def __init__(
        self,
        root: str,
        image_set: str = "trainaug",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        file_set: Optional[List[str]] = None,
        return_masks: bool = False
    ):
        # either transform and target_transform should be passed or only transforms
        super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        self.root = root
        self.return_masks = return_masks

        # Mode detection
        self._is_tar = _looks_like_tar_path(self.root)
        self._tar = None  # lazily opened per worker

        self.images, self.masks = self._collect_data(file_set)
        print(f"Found {len(self.images)} images and {len(self.masks)} masks in {self.root}")

    def _collect_data(self, file_set=None) -> Tuple[List[str], List[str]]:
        # Decide segmentation folder by split
        if self.image_set in ("trainaug", "train"):
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")

        if not self._is_tar:
            # Directory mode (original behavior with a tad more robustness)
            image_dir = os.path.join(self.root, 'images')
            seg_dir = os.path.join(self.root, seg_folder)

            if not (os.path.isdir(self.root) and os.path.isdir(image_dir) and os.path.isdir(seg_dir)):
                raise RuntimeError('Dataset not found or corrupted.')

            if file_set is None:
                images = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith('.jpg')]
                masks = [os.path.join(seg_dir, f) for f in sorted(os.listdir(seg_dir)) if f.lower().endswith('.png')]
                images, masks = _pair_by_stem_dir(image_dir, seg_dir, images, masks)
            else:
                images = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(file_set)]
                masks = [os.path.join(seg_dir, f'{f}.png') for f in sorted(file_set)]
                # ensure they exist
                assert all(Path(f).is_file() for f in images), "Some requested images are missing"
                assert all(Path(f).is_file() for f in masks), "Some requested masks are missing"

            return images, masks

        # Tar mode
        # Expect internal paths like:
        # VOCSegmentation/images/*.jpg
        # VOCSegmentation/SegmentationClass/*.png
        # VOCSegmentation/SegmentationClassAug/*.png
        img_prefixes = [
            'VOCSegmentation/images/',
            './VOCSegmentation/images/',
        ]
        seg_prefixes = [
            f'VOCSegmentation/{seg_folder}/',
            f'./VOCSegmentation/{seg_folder}/',
            # be tolerant to archives missing the trailing slash in the stored prefix
            f'VOCSegmentation/{seg_folder}',
            f'./VOCSegmentation/{seg_folder}',
        ]

        # Scan member names once to map stems->members
        with tarfile.open(self.root, 'r:*') as t:
            img_members, seg_members = [], []
            for m in t.getmembers():
                if not m.isreg():
                    continue
                p = _norm_tar_path(m.name)
                if p.lower().endswith('.jpg') and any(p.startswith(pref) for pref in img_prefixes):
                    img_members.append(p)
                elif p.lower().endswith('.png') and any(p.startswith(pref) for pref in seg_prefixes):
                    seg_members.append(p)

        img_members.sort()
        seg_members.sort()

        if file_set is None:
            images, masks = _pair_by_stem_tar(img_members, seg_members)
        else:
            wanted = sorted(file_set)
            img_map = {stem_from_path(p): p for p in img_members}
            seg_map = {stem_from_path(p): p for p in seg_members}
            images, masks = [], []
            for s in wanted:
                im = img_map.get(s)
                mk = seg_map.get(s)
                if im is not None and mk is not None:
                    images.append(im)
                    masks.append(mk)
            # (optional) enforce strictness:
            # missing = [s for s in wanted if s not in img_map or s not in seg_map]
            # if missing: raise FileNotFoundError(f"Missing items in tar: {missing}")

        return images, masks

    def _lazy_open_tar(self):
        if self._tar is None and self._is_tar:
            self._tar = tarfile.open(self.root, 'r:*')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not self._is_tar:
            img = Image.open(self.images[index]).convert('RGB')
            if self.image_set == "val":
                mask = Image.open(self.masks[index])
                if self.transforms:
                    img, mask = self.transforms(img, mask)
                return img, mask
            elif "train" in self.image_set:
                if self.transforms:
                    if self.return_masks:
                        mask = Image.open(self.masks[index])
                        res = self.transforms(img, mask)
                    else:
                        res = self.transforms(img)
                    return res
                return img

        # Tar mode
        self._lazy_open_tar()
        assert self._tar is not None

        with self._tar.extractfile(self.images[index]) as f_img:
            img = Image.open(io.BytesIO(f_img.read())).convert('RGB')

        if self.image_set == "val":
            with self._tar.extractfile(self.masks[index]) as f_mk:
                mask = Image.open(io.BytesIO(f_mk.read()))
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask

        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    with self._tar.extractfile(self.masks[index]) as f_mk:
                        mask = Image.open(io.BytesIO(f_mk.read()))
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)

    # Make dataset pickle-friendly for multiprocess DataLoader
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_tar'] = None
        return state

    def __del__(self):
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass


# -------------------------
# Helpers
# -------------------------

def _looks_like_tar_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz') \
        or lower.endswith('.tar.bz2') or lower.endswith('.tbz2') or lower.endswith('.tar.xz') or lower.endswith('.txz')


def _norm_tar_path(p: str) -> str:
    return p[2:] if p.startswith('./') else p


def stem_from_path(p: str) -> str:
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return stem


def _pair_by_stem_dir(image_dir: str, seg_dir: str,
                      image_paths: List[str], seg_paths: List[str]) -> Tuple[List[str], List[str]]:
    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_paths}
    seg_map = {os.path.splitext(os.path.basename(f))[0]: f for f in seg_paths}
    common = sorted(set(img_map.keys()) & set(seg_map.keys()))
    images = [img_map[s] for s in common]
    masks = [seg_map[s] for s in common]
    return images, masks


def _pair_by_stem_tar(img_members: List[str], seg_members: List[str]) -> Tuple[List[str], List[str]]:
    img_map = {stem_from_path(p): p for p in img_members}
    seg_map = {stem_from_path(p): p for p in seg_members}
    common = sorted(set(img_map.keys()) & set(seg_map.keys()))
    images = [img_map[s] for s in common]
    masks = [seg_map[s] for s in common]
    return images, masks
