import numpy as np
import os
import torch
import pickle
import pytorch_lightning as pl

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import data.ade20k.utils_ade20k as utils

from typing import Optional

from PIL import Image
from PIL.Image import NEAREST


class Ade20kPartsDataModule(pl.LightningDataModule):

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 val_target_transforms):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_target_transforms = val_target_transforms

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            train_len = self.get_train_dataset_size()
            train_start_index = 0
            val_len = self.get_val_dataset_size()
            self.val = Ade20KPartsDataset(self.root, train_len + val_len, train_start_index,
                                          img_transform=self.val_transforms,
                                          mask_transform=self.val_target_transforms)
            self.train = self.val
            print(f"Val size {len(self.val)}")
        else:
            raise NotImplementedError("Unlabelled NEON doesn't have a dedicated val/test set.")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def get_train_dataset_size(self):
        return 25258

    def get_val_dataset_size(self):
        return 2000
    
    def get_num_classes(self):
        return 111


class Ade20kDataModule(pl.LightningDataModule):

    def __init__(self,
                 root,
                 train_transforms,
                 val_transforms,
                 shuffle,
                 num_workers,
                 batch_size,
                 val_file_set=None,
                 train_file_set=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.val_file_set = val_file_set
        self.train_file_set = train_file_set

    def setup(self, stage: Optional[str] = None):
        self.val = ADE20K(self.root, self.val_transforms, split='val', file_set=self.val_file_set)
        self.train = ADE20K(self.root, self.train_transforms, split='train', file_set=self.train_file_set)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.train)

    def get_val_dataset_size(self):
        return len(self.val)
    
    def get_num_classes(self):
        return 151


class ADE20K(Dataset):
    split_to_dir = {
        'train': 'training',
        'val': 'validation'
    }

    def __init__(self, root, transforms, split='train', skip_other_class=False, file_set=None):
        super().__init__()
        self.transforms = transforms
        self.split = split
        self.root = root
        self.skip_other_class = skip_other_class
        self.file_set = file_set

        # Collect the data
        self.data = self.collect_data()

    def collect_data(self):
        # Get the image and annotation dirs
        image_dir = os.path.join(self.root, f'images/{self.split_to_dir[self.split]}')
        annotation_dir = os.path.join(self.root, f'annotations/{self.split_to_dir[self.split]}')

        # Collect the filepaths
        if self.file_set is None:
            image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            annotation_paths = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir))]
        else:
            image_paths = [os.path.join(image_dir, f'{f}.jpg') for f in sorted(self.file_set)]
            annotation_paths = [os.path.join(annotation_dir, f'{f}.png') for f in sorted(self.file_set)]

        data = list(zip(image_paths, annotation_paths))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the  paths
        image_path, annotation_path = self.data[index]

        # Load
        image = Image.open(image_path).convert("RGB")
        target = Image.open(annotation_path)

        # Augment
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            # Convert unwanted class to the class to skip
            # which in our case we always skip the class of 255
        else:
            target = F.pil_to_tensor(target)

        if self.skip_other_class == True:
            target = target * 255.0
            target[target.type(torch.int64)==0]=255.0
            target /= 255.0

        if self.transforms is None:
            target = F.to_pil_image(target)
        
        return image, target

class Ade20KPartsDataset(Dataset):
    def __init__(self, root, val_len, val_start_index, mask_transform=None, img_transform=None):
        self.root = root
        with open(os.path.join(self.root, "ADE20K_2021_17_01/index_ade20k.pkl"), 'rb') as f:
            index_ade20k = pickle.load(f)

        # get val idx  and img idst
        self.idx = self.construct_index(index_ade20k, val_start_index, val_len)
        # select images that have object part annotations and are street scenes
        part_img_ids = np.sum(self.idx["objectIsPart"], axis=0).nonzero()[0]
        street_img_idx = (np.array(self.idx["scene"]) == "/street").nonzero()[0]
        print(f"Found {len(street_img_idx)} street images")
        self.image_ids = list(set(part_img_ids).intersection(set(street_img_idx)))
        print(f"Found {len(self.image_ids)} street images with parts annotations")
        # construct part id map
        # all part ids found during iteration through whole dataset
        all_part_ids = [50, 51, 54, 83, 101, 112, 135, 136, 144, 175, 184, 211, 213, 277, 320, 495, 543, 544, 580, 582,
                        626, 665, 772, 774, 776, 777, 781, 783, 785, 840, 859, 860, 876, 890, 904, 909, 934, 938, 1062,
                        1063, 1072, 1081, 1140, 1145, 1156, 1180, 1206, 1212, 1213, 1249, 1259, 1277, 1279, 1280, 1395,
                        1397, 1412, 1428, 1429, 1430, 1431, 1439, 1470, 1540, 1541, 1564, 1882, 1883, 1936, 1951, 1957,
                        2052, 2067, 2103, 2117, 2118, 2119, 2120, 2122, 2130, 2155, 2156, 2164, 2190, 2346, 2370, 2371,
                        2376, 2379, 2421, 2529, 2564, 2567, 2570, 2700, 2742, 2820, 2828, 2855, 2884, 2940, 2978, 3035,
                        3050, 3054, 3056, 3057, 3063, 3137, 3153, 3154]
        print(f"Found {len(all_part_ids)} parts")
        self.part_id_map = {part_id: i for i, part_id in enumerate(all_part_ids)}
        self.part_id_map[0] = 255 # map no parts pixels to ignore index
        self.mask_transform = mask_transform
        self.img_transform = img_transform

    def construct_index(self, index_ade20k, start_idx, length):
        index_ade20k["filename"] = index_ade20k["filename"][start_idx:start_idx+length]
        index_ade20k["folder"] = index_ade20k["folder"][start_idx:start_idx+length]
        index_ade20k["objectIsPart"] = index_ade20k["objectIsPart"][:, start_idx: start_idx + length]
        index_ade20k["objectPresence"] = index_ade20k["objectPresence"][:, start_idx: start_idx + length]
        index_ade20k["scene"] = index_ade20k["scene"][start_idx:start_idx + length]
        return index_ade20k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        full_file_path = os.path.join(self.idx['folder'][i], self.idx['filename'][i])
        info = utils.loadAde20K(os.path.join(self.root, full_file_path))
        img = Image.open(info['img_name'])
        img = self.img_transform(img) # normalize and resize image
        parts_mask = torch.from_numpy(info['partclass_mask'][0]).unsqueeze(0).float() # only keep parts not parts of parts
        parts_mask = self.mask_transform(parts_mask) # resize parts mask

        # zero index and linearize class part classes. Assign non parts ignore index
        linearized_mask = parts_mask.clone()
        for part_id in torch.unique(parts_mask):
            linearized_mask[parts_mask == part_id] = self.part_id_map[part_id.item()]
        linearized_mask /= 255

        return img, linearized_mask
