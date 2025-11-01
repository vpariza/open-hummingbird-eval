import io
import json
import os
import tarfile
from typing import List, Optional, Callable, Tuple, Any

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CocoDataModule(pl.LightningDataModule):

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 data_dir: str,
                 train_transforms,
                 val_transforms,
                 mask_type: str = None,
                 shuffle: bool = True,
                 drop_last=False,
                 train_file_set: List[str] = None,
                 val_file_set: List[str] = None):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_file_set = train_file_set
        self.val_file_set = val_file_set
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.mask_type = mask_type
        self.coco_train = None
        self.coco_val = None
        self.drop_last = drop_last

    def __len__(self):
        return len(self.train_file_set)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        self.coco_train = COCOSegmentation(self.data_dir,
                                           self.mask_type,
                                           image_set="train",
                                           transforms=self.train_transforms,
                                           file_set=self.train_file_set)
        self.coco_val = COCOSegmentation(self.data_dir,
                                         self.mask_type,
                                         image_set="val",
                                         transforms=self.val_transforms,
                                         file_set=self.val_file_set)

        print(f"Train size {len(self.coco_train)}")
        print(f"Val size {len(self.coco_val)}")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.coco_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.coco_train)

    def get_val_dataset_size(self):
        return len(self.coco_val)

    def get_num_classes(self):
        if self.mask_type == "thing":
            return 12
        else:
            return 15


class COCOSegmentation(Dataset):

    def __init__(
        self,
        root: str,
        mask_type: str,
        image_set: str = "train",
        transforms: Optional[Callable] = None,
        file_set: List[str] = None,
    ):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.image_set = image_set
        self.file_set = file_set
        self.mask_type = mask_type
        assert self.image_set in ["train", "val"]
        assert mask_type in ["stuff", "thing"]

        # Mode detection
        self._is_tar = _looks_like_tar_path(self.root)
        self._tar = None  # lazily opened in workers

        # Set mask folder and category JSON (kept identical to your original logic)
        if mask_type == "thing":
            seg_folder = "annotations/{}2017/"
            json_file = "annotations/panoptic_annotations/panoptic_val2017.json"
        elif mask_type == "stuff":
            seg_folder = "annotations/stuff_annotations/stuff_{}2017_pixelmaps/"
            json_file = "annotations/stuff_annotations/stuff_val2017.json"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        self._seg_folder = seg_folder.format(image_set)
        self._json_file = json_file

        # Load categories -> coarse id map
        self.cat_id_map = self._load_category_map()

        # Build image/mask lists (paths in dir mode, member names in tar mode)
        self.images, self.masks = self._collect_files()

    def _lazy_open_tar(self):
        if self._tar is None and self._is_tar:
            self._tar = tarfile.open(self.root, 'r:*')

    def _load_category_map(self):
        if not self._is_tar:
            with open(os.path.join(self.root, self._json_file)) as f:
                an_json = json.load(f)
        else:
            # load JSON from tar
            json_candidates = [self._json_file, f'./{self._json_file}']
            with tarfile.open(self.root, 'r:*') as t:
                member = None
                for cand in json_candidates:
                    member = t.getmember(cand) if cand in t.getnames() else member
                if member is None:
                    # fallback: linear search to be tolerant of odd roots
                    for n in t.getnames():
                        n_norm = _norm_tar_path(n)
                        if n_norm == self._json_file:
                            member = t.getmember(n)
                            break
                if member is None:
                    raise RuntimeError(f"Could not find JSON '{self._json_file}' in tar")
                with t.extractfile(member) as jf:
                    an_json = json.load(jf)

        all_cat = an_json['categories']
        if self.mask_type == "thing":
            all_thing_cat_sup = set(cat_dict["supercategory"] for cat_dict in all_cat if cat_dict.get("isthing", 0) == 1)
            super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(all_thing_cat_sup))}
            cat_id_map = {}
            for cat_dict in all_cat:
                if cat_dict.get("isthing", 0) == 1:
                    cat_id_map[cat_dict["id"]] = super_cat_to_id[cat_dict["supercategory"]]
                else:
                    cat_id_map[cat_dict["id"]] = 255
        else:
            super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
            if "other" in super_cats:
                super_cats.remove("other")
            super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
            super_cat_to_id["other"] = 255
            cat_id_map = {cat_dict['id']: super_cat_to_id.get(cat_dict['supercategory'], 255) for cat_dict in all_cat}
        return cat_id_map

    def _collect_files(self) -> Tuple[List[str], List[str]]:
        image_dir = os.path.join(self.root, "images", f"{self.image_set}2017")
        annotation_dir = os.path.join(self.root, self._seg_folder)

        if not self._is_tar:
            if not (os.path.isdir(annotation_dir) and os.path.isdir(image_dir)):
                print(annotation_dir)
                print(image_dir)
                raise RuntimeError('Dataset not found or corrupted.')

            if self.file_set is None:
                imgs = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.lower().endswith('.jpg')]
                msks = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir)) if f.lower().endswith('.png')]
                return _pair_by_stem_dir(imgs, msks)
            else:
                base_set = [f.replace(".jpg", "").replace(".png", "") for f in self.file_set]
                imgs = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(base_set)]
                msks = [os.path.join(annotation_dir, f'{f}.png') for f in sorted(base_set)]
                return imgs, msks

        # ---- Tar mode ----
        # Expect internal members:
        #   images/{split}2017/*.jpg
        #   annotations/{split-specific}/*.png
        img_prefixes = [
            f'images/{self.image_set}2017/',
            f'./images/{self.image_set}2017/',
        ]
        seg_prefixes = [
            _norm_tar_path(self._seg_folder),
            _norm_tar_path(f'./{self._seg_folder}'),
        ]

        img_members, seg_members = [], []
        with tarfile.open(self.root, 'r:*') as t:
            for m in t.getmembers():
                if not m.isreg():
                    continue
                p = _norm_tar_path(m.name)
                if p.lower().endswith('.jpg') and any(p.startswith(pref) for pref in img_prefixes):
                    if self.file_set is None or stem_from_path(p) in self._normalized_file_set():
                        img_members.append(p)
                elif p.lower().endswith('.png') and any(p.startswith(pref) for pref in seg_prefixes):
                    if self.file_set is None or stem_from_path(p) in self._normalized_file_set():
                        seg_members.append(p)

        img_members.sort()
        seg_members.sort()
        return _pair_by_stem_tar(img_members, seg_members)

    def _normalized_file_set(self):
        if self.file_set is None:
            return None
        return set(f.replace(".jpg", "").replace(".png", "") for f in self.file_set)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not self._is_tar:
            img = Image.open(self.images[index]).convert('RGB')
            mask = Image.open(self.masks[index])
        else:
            self._lazy_open_tar()
            assert self._tar is not None
            with self._tar.extractfile(self.images[index]) as f_img:
                img = Image.open(io.BytesIO(f_img.read())).convert('RGB')
            with self._tar.extractfile(self.masks[index]) as f_mk:
                mask = Image.open(io.BytesIO(f_mk.read()))

        if self.transforms:
            img, mask = self.transforms(img, mask)

        if self.mask_type == "stuff":
            # move stuff labels from {0} U [92, 183] to [0,15] and {255}; 255 == {0, 183}
            mask *= 255
            assert torch.max(mask).item() <= 183
            mask[mask == 0] = 183  # [92, 183]
            assert torch.min(mask).item() >= 92
            for cat_id in torch.unique(mask):
                mask[mask == cat_id] = self.cat_id_map[int(cat_id.item())]
            assert torch.max(mask).item() <= 255
            assert torch.min(mask).item() >= 0
            mask /= 255
            return img, mask

        elif self.mask_type == "thing":
            mask *= 255
            mask[mask == 0] = 200  # map unlabelled to stuff
            merged_mask = mask.clone()
            for cat_id in torch.unique(mask):
                cid = int(cat_id.item())
                if cid in self.cat_id_map and cid <= 200:
                    merged_mask[mask == cat_id] = self.cat_id_map[cid]  # [0, 11] + {255}
                else:
                    merged_mask[mask == cat_id] = 255
            assert torch.max(merged_mask).item() <= 255
            assert torch.min(merged_mask).item() >= 0
            merged_mask /= 255
            return img, merged_mask

        return img, mask

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


def _pair_by_stem_dir(images: List[str], masks: List[str]) -> Tuple[List[str], List[str]]:
    img_map = {stem_from_path(f): f for f in images}
    msk_map = {stem_from_path(f): f for f in masks}
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    return [img_map[s] for s in common], [msk_map[s] for s in common]


def _pair_by_stem_tar(img_members: List[str], seg_members: List[str]) -> Tuple[List[str], List[str]]:
    img_map = {stem_from_path(p): p for p in img_members}
    seg_map = {stem_from_path(p): p for p in seg_members}
    common = sorted(set(img_map.keys()) & set(seg_map.keys()))
    return [img_map[s] for s in common], [seg_map[s] for s in common]
