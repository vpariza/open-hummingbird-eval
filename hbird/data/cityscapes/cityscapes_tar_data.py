import os
import io
import tarfile
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import torch


class Cityscapes(Dataset):
    def __init__(self, root, transforms, split='train', file_set: Optional[List[str]] = None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split

        # Mode detection
        self._is_tar = _looks_like_tar_path(self.root)
        self._tar = None  # lazily opened per worker

        if not os.path.exists(self.root):
            raise AssertionError("Please setup the dataset properly")

        # Collect pairs (paths or tar member names)
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split, file_set, self._is_tar)
        assert len(self.images) == len(self.mask_paths)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")

        # Class mapping (unchanged)
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.inv_index = 255
        inv = self.inv_index
        self._key = np.array([inv, inv, inv, inv, inv, inv,
                              inv, inv, 0, 1, inv, inv,
                              2, 3, 4, inv, inv, inv,
                              5, inv, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              inv, inv, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def _lazy_open_tar(self):
        if self._tar is None and self._is_tar:
            self._tar = tarfile.open(self.root, 'r:*')

    def __getitem__(self, index):
        if not self._is_tar:
            # Directory mode
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.mask_paths[index])
        else:
            # Tar mode
            self._lazy_open_tar()
            assert self._tar is not None
            with self._tar.extractfile(self.images[index]) as f_img:
                image = Image.open(io.BytesIO(f_img.read())).convert('RGB')
            with self._tar.extractfile(self.mask_paths[index]) as f_mk:
                target = Image.open(io.BytesIO(f_mk.read()))

        target = self._mask_transform(target).float() / 255.0
        target = F.to_pil_image(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_tar'] = None  # file handles aren't picklable
        return state

    def __del__(self):
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass


def _get_city_pairs(folder_or_tar: str,
                    split: str = 'train',
                    file_set: Optional[List[str]] = None,
                    is_tar: bool = False) -> Tuple[List[str], List[str]]:
    """
    Returns two aligned lists of (images, masks). Each element is either:
      - a filesystem path (directory mode), or
      - a tar member name (tar mode).
    file_set: optional iterable of base names BEFORE the '_leftImg8bit.png' suffix.
              e.g., 'frankfurt_000000_000294'  (no extension/suffix/city)
    """
    if split not in ('train', 'val', 'trainval'):
        raise ValueError(f"Invalid split: {split}")

    if not is_tar:
        # ------------- Directory mode (original behavior preserved) -------------
        def get_path_pairs(img_folder, mask_folder, file_set_local=None):
            img_paths, mask_paths = [], []
            allowed = set(file_set_local) if file_set_local is not None else None
            for root, _, files in os.walk(img_folder):
                for filename in files:
                    if not filename.endswith('.png'):
                        continue
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    base_name = filename.split("_leftImg8bit.png")[0]
                    if allowed is not None and base_name not in allowed:
                        continue
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
            print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
            return img_paths, mask_paths

        if split in ('train', 'val'):
            base = folder_or_tar
            img_folder = os.path.join(base, f'leftImg8bit/{split}')
            mask_folder = os.path.join(base, f'gtFine/{split}')
            return get_path_pairs(img_folder, mask_folder, file_set)
        else:
            base = folder_or_tar
            train_img_folder = os.path.join(base, 'leftImg8bit/train')
            train_mask_folder = os.path.join(base, 'gtFine/train')
            val_img_folder = os.path.join(base, 'leftImg8bit/val')
            val_mask_folder = os.path.join(base, 'gtFine/val')
            train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
            return train_img_paths + val_img_paths, train_mask_paths + val_mask_paths

    # ------------- Tar mode -------------
    # Expect internal names like:
    #   cityscapes/leftImg8bit/{split}/{city}/*_leftImg8bit.png
    #   cityscapes/gtFine/{split}/{city}/*_gtFine_labelIds.png
    prefixes = {
        'img': [f'cityscapes/leftImg8bit/', f'./cityscapes/leftImg8bit/'],
        'gt':  [f'cityscapes/gtFine/', f'./cityscapes/gtFine/'],
    }

    def scan_split_from_tar(tar_path: str, split_name: str, file_set_local=None) -> Tuple[List[str], List[str]]:
        allowed = set(file_set_local) if file_set_local is not None else None
        img_members, mask_members = [], []
        with tarfile.open(tar_path, 'r:*') as t:
            for m in t.getmembers():
                if not m.isreg():
                    continue
                p = _norm_tar_path(m.name)
                # quick split filtering
                if f'/leftImg8bit/{split_name}/' in p and p.endswith('_leftImg8bit.png'):
                    if any(p.startswith(pref) for pref in prefixes['img']):
                        base = _base_from_left(p)
                        if allowed is None or base in allowed:
                            img_members.append(p)
                elif f'/gtFine/{split_name}/' in p and p.endswith('_gtFine_labelIds.png'):
                    if any(p.startswith(pref) for pref in prefixes['gt']):
                        base = _base_from_label(p)
                        if allowed is None or base in allowed:
                            mask_members.append(p)
        img_members.sort()
        mask_members.sort()
        # Pair by base (before suffix)
        images, masks = _pair_by_base(img_members, mask_members)
        print(f"Found {len(images)} images in tar split '{split_name}'")
        return images, masks

    if split in ('train', 'val'):
        return scan_split_from_tar(folder_or_tar, split, file_set)
    else:
        tr_i, tr_m = scan_split_from_tar(folder_or_tar, 'train', None)
        va_i, va_m = scan_split_from_tar(folder_or_tar, 'val', None)
        return tr_i + va_i, tr_m + va_m


class CityscapesDataModule(pl.LightningDataModule):
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
        self.val = Cityscapes(self.root, self.val_transforms, split='val', file_set=self.val_file_set)
        self.train = Cityscapes(self.root, self.train_transforms, split='train', file_set=self.train_file_set)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.train)

    def get_val_dataset_size(self):
        return len(self.val)

    def get_num_classes(self):
        return 19


# -------------------------
# Helpers
# -------------------------

def _looks_like_tar_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz') \
        or lower.endswith('.tar.bz2') or lower.endswith('.tbz2') or lower.endswith('.tar.xz') or lower.endswith('.txz')


def _norm_tar_path(p: str) -> str:
    return p[2:] if p.startswith('./') else p


def _base_from_left(p: str) -> str:
    # returns base name before '_leftImg8bit.png'
    name = os.path.basename(p)
    return name[:-len('_leftImg8bit.png')] if name.endswith('_leftImg8bit.png') else os.path.splitext(name)[0]


def _base_from_label(p: str) -> str:
    # returns base name before '_gtFine_labelIds.png'
    name = os.path.basename(p)
    return name[:-len('_gtFine_labelIds.png')] if name.endswith('_gtFine_labelIds.png') else os.path.splitext(name)[0]


def _pair_by_base(img_members: List[str], mask_members: List[str]) -> Tuple[List[str], List[str]]:
    img_map = {_base_from_left(p): p for p in img_members}
    mask_map = {_base_from_label(p): p for p in mask_members}
    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    images = [img_map[s] for s in common]
    masks = [mask_map[s] for s in common]
    return images, masks
