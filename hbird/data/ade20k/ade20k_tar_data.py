import os
import io
import tarfile
from typing import Optional, List, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import functional as TF  # for pil_to_tensor / to_pil_image


class Ade20kDataModule(pl.LightningDataModule):
    """
    Public API unchanged. Pass `root` as either:
      - directory containing ade20k/images/... and ade20k/annotations/...
      - OR a .tar file with the same internal structure.
    """

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 drop_last=False,
                 train_file_set=None,
                 val_file_set=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_file_set = val_file_set
        self.train_file_set = train_file_set
        self.drop_last = drop_last

    def setup(self, stage: Optional[str] = None):
        # Works for both directory roots and .tar roots
        self.val = ADE20K(self.root, self.val_transforms, split='val', file_set=self.val_file_set)
        self.train = ADE20K(self.root, self.train_transforms, split='train', file_set=self.train_file_set)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.drop_last,
                          pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.train)

    def get_val_dataset_size(self):
        return len(self.val)

    def get_num_classes(self):
        return 151


class ADE20K(Dataset):
    """
    Drop-in replacement that supports:
      • Directory layout: <root>/images/training/*.jpg, <root>/annotations/training/*.png, etc.
      • Tar layout: a .tar whose internal paths mirror the same layout:
          ade20k/images/training/*.jpg
          ade20k/images/validation/*.jpg
          ade20k/annotations/training/*.png
          ade20k/annotations/validation/*.png
    """
    split_to_dir = {
        'train': 'training',
        'val': 'validation'
    }

    def __init__(self,
                 root: str,
                 transforms,
                 split: str = 'train',
                 skip_other_class: bool = False,
                 file_set: Optional[List[str]] = None):
        super().__init__()
        self.transforms = transforms
        self.split = split
        self.root = root
        self.skip_other_class = skip_other_class

        # mode detection (folder vs tar)
        self._is_tar = _looks_like_tar_path(self.root)

        # Build the (image_ref, ann_ref) pairs.
        # In dir mode: refs are absolute file paths.
        # In tar mode: refs are *tar internal member names* (strings).
        self.data: List[Tuple[str, str]] = self._collect_data(file_set)

        # Tar file handle is opened lazily (per worker) on first __getitem__
        self._tar = None  # type: Optional[tarfile.TarFile]

    def _collect_data(self, file_set: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        split_dir = self.split_to_dir[self.split]
        if not self._is_tar:
            # Directory mode (original behavior)
            image_dir = os.path.join(self.root, f'images/{split_dir}')
            annotation_dir = os.path.join(self.root, f'annotations/{split_dir}')

            if file_set is None:
                image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
                ann_files = sorted([f for f in os.listdir(annotation_dir) if f.lower().endswith('.png')])
                # rely on aligned naming; but to be robust, match by stem
                data = _pair_by_stem_dir(image_dir, annotation_dir, image_files, ann_files)
            else:
                image_paths = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(file_set)]
                annotation_paths = [os.path.join(annotation_dir, f'{f}.png') for f in sorted(file_set)]
                data = list(zip(image_paths, annotation_paths))
            return data

        # Tar mode
        # We don't open the tar yet; we only scan member names once here for mapping.
        tar_path = self.root

        # Build lists of internal member names for images and annotations
        # Accept both "ade20k/..." and "./ade20k/..." internal roots.
        img_prefixes = [
            f'ade20k/images/{split_dir}/',
            f'./ade20k/images/{split_dir}/'
        ]
        ann_prefixes = [
            f'ade20k/annotations/{split_dir}/',
            f'./ade20k/annotations/{split_dir}/'
        ]

        # We need to read member names without fully loading the tar.
        with tarfile.open(tar_path, 'r:*') as t:
            img_members = []
            ann_members = []
            for m in t.getmembers():
                if not m.isreg():  # skip non-regular files
                    continue
                p = _norm_tar_path(m.name)
                if p.lower().endswith('.jpg') and any(p.startswith(pref) for pref in img_prefixes):
                    img_members.append(p)
                elif p.lower().endswith('.png') and any(p.startswith(pref) for pref in ann_prefixes):
                    ann_members.append(p)

        img_members.sort()
        ann_members.sort()

        if file_set is None:
            # Pair by stem (robust even if tar order differs)
            return _pair_by_stem_tar(img_members, ann_members)
        else:
            wanted = sorted(file_set)
            img_map = {stem_from_path(p): p for p in img_members}
            ann_map = {stem_from_path(p): p for p in ann_members}
            pairs: List[Tuple[str, str]] = []
            for stem in wanted:
                img_ref = img_map.get(stem)
                ann_ref = ann_map.get(stem)
                if img_ref is None or ann_ref is None:
                    # If a requested item is missing, skip it (or raise if you prefer strict)
                    continue
                pairs.append((img_ref, ann_ref))
            return pairs

    def __len__(self):
        return len(self.data)

    def _lazy_open_tar(self):
        if self._tar is None and self._is_tar:
            # Open at first access (per worker)
            self._tar = tarfile.open(self.root, 'r:*')

    def __getitem__(self, index: int):
        image_ref, annotation_ref = self.data[index]

        if not self._is_tar:
            # Directory mode
            image = Image.open(image_ref).convert("RGB")
            target = Image.open(annotation_ref)
        else:
            # Tar mode
            self._lazy_open_tar()
            assert self._tar is not None
            # Access members by name (fast lookup via internal index)
            with self._tar.extractfile(image_ref) as f_img:
                image = Image.open(io.BytesIO(f_img.read())).convert("RGB")
            with self._tar.extractfile(annotation_ref) as f_ann:
                target = Image.open(io.BytesIO(f_ann.read()))

        # Apply transforms or default tensor conversion path
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Fallback to tensor conversions (keep parity with original intent)
            target = TF.pil_to_tensor(target)

        if self.skip_other_class is True:
            # Map "other" to 255, retain others
            target = target * 255.0
            target[target.type(torch.int64) == 0] = 255.0
            target = target / 255.0

        if self.transforms is None:
            # Bring target back to PIL if no transforms were applied
            target = TF.to_pil_image(target)

        return image, target

    # Make dataset pickle-friendly for multiprocess DataLoader
    def __getstate__(self):
        state = self.__dict__.copy()
        # Tar file handles cannot be pickled
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
    # Normalize tar internal paths; remove redundant './'
    if p.startswith('./'):
        return p[2:]
    return p


def stem_from_path(p: str) -> str:
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return stem


def _pair_by_stem_dir(image_dir: str, ann_dir: str,
                      image_files: List[str], ann_files: List[str]) -> List[Tuple[str, str]]:
    img_map = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in image_files}
    ann_map = {os.path.splitext(f)[0]: os.path.join(ann_dir, f) for f in ann_files}
    common = sorted(set(img_map.keys()) & set(ann_map.keys()))
    return [(img_map[s], ann_map[s]) for s in common]


def _pair_by_stem_tar(img_members: List[str], ann_members: List[str]) -> List[Tuple[str, str]]:
    img_map = {stem_from_path(p): p for p in img_members}
    ann_map = {stem_from_path(p): p for p in ann_members}
    common = sorted(set(img_map.keys()) & set(ann_map.keys()))
    return [(img_map[s], ann_map[s]) for s in common]
