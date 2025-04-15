import json
import pytorch_lightning as pl
import os
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Callable, Tuple, Any


class CocoDataModule(pl.LightningDataModule):

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 data_dir: str,
                 train_transforms,
                 val_transforms,
                 mask_type: str = None,
                 shuffle: bool = True,
                 train_file_set: List[str]=None,
                 val_file_set: List[str]=None):
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
                          drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.coco_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

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

        # Set mask folder depending on mask_type
        if mask_type == "thing":
            seg_folder = "annotations/{}2017/"
            json_file = "annotations/panoptic_annotations/panoptic_val2017.json"
        elif mask_type == "stuff":
            seg_folder = "annotations/stuff_annotations/stuff_{}2017_pixelmaps/"
            json_file = "annotations/stuff_annotations/stuff_val2017.json"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_folder = seg_folder.format(image_set)

        # Load categories to category to id map for merging to coarse categories
        with open(os.path.join(root, json_file)) as f:
            an_json = json.load(f)
            all_cat = an_json['categories']
            if mask_type == "thing":
                all_thing_cat_sup = set(cat_dict["supercategory"] for cat_dict in all_cat if cat_dict["isthing"] == 1)
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(all_thing_cat_sup))}
                self.cat_id_map = {}
                for cat_dict in all_cat:
                    if cat_dict["isthing"] == 1:
                        self.cat_id_map[cat_dict["id"]] = super_cat_to_id[cat_dict["supercategory"]]
                    elif cat_dict["isthing"] == 0:
                        self.cat_id_map[cat_dict["id"]] = 255
            else:
                super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
                super_cats.remove("other")  # remove others from prediction targets as this is not semantic
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
                super_cat_to_id["other"] = 255  # ignore_index for CE
                self.cat_id_map = {cat_dict['id']: super_cat_to_id[cat_dict['supercategory']] for cat_dict in all_cat}

        # Get images and masks fnames
        annotation_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, "images", f"{image_set}2017")
        if not os.path.isdir(annotation_dir) or not os.path.isdir(image_dir):
            print(annotation_dir)
            print(image_dir)
            raise RuntimeError('Dataset not found or corrupted.')

        # Collect the filepaths
        if self.file_set is None:
            self.images = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            self.masks = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir))]
        else:
            self.file_set = [f.replace(".jpg","").replace(".png","") for f in self.file_set]
            self.images = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(self.file_set)]
            self.masks = [os.path.join(annotation_dir, f'{f}.png') for f in sorted(self.file_set)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.transforms:
            img, mask = self.transforms(img, mask)

        if self.mask_type == "stuff":
            # move stuff labels from {0} U [92, 183] to [0,15] and [255] with 255 == {0, 183}
            # (183 is 'other' and 0 is things)
            mask *= 255
            assert torch.max(mask).item() <= 183
            mask[mask == 0] = 183  # [92, 183]
            assert torch.min(mask).item() >= 92
            for cat_id in torch.unique(mask):
                mask[mask == cat_id] = self.cat_id_map[cat_id.item()]

            assert torch.max(mask).item() <= 255
            assert torch.min(mask).item() >= 0
            mask /= 255
            return img, mask
        elif self.mask_type == "thing":
            mask *= 255
            # assert torch.max(mask[mask!=255]).item() <= 200
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

