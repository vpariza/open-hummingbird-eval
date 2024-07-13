import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from src.utils.transforms import get_default_val_transforms

import torch
from abc import ABC, abstractmethod
import os
import pandas as pd

import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image


class Dataset(torch.nn.Module):
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @abstractmethod
    def get_val_dataset(self):
        pass
    
    @abstractmethod
    def get_num_classes(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass



class NYUDataset(Dataset):
    def __init__(
            self,
            root: str,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        # super(VOCDataset, self).__init__(root, transforms, transform, target_transform)
        super(NYUDataset, self).__init__()
        ## check if data is not part of root

        if "data" not in os.listdir(root):
            self.root = os.path.join(root, "data")
        else:
            self.root = root

        if image_set == "train":
            csv_path = os.path.join(root, 'data/nyu2_train.csv')
        elif image_set == "val":
            csv_path = os.path.join(root, 'data/nyu2_test.csv')
        
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.return_masks = return_masks    
        file_paths = pd.read_csv(csv_path)
        self.RGB_paths = [os.path.join(self.root, img) for img in file_paths.iloc[:, 0]]
        self.depth_paths = [os.path.join(self.root, img) for img in file_paths.iloc[:, 1]]

        assert len(self.RGB_paths) == len(self.depth_paths) 



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # print(self.RGB_paths[index])
        # print(self.depth_paths[index])
        img = Image.open(self.RGB_paths[index])
        depth = Image.open(self.depth_paths[index])

        if self.image_set == "val":
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                img, depth = self.transforms(img, depth)
            return img, depth
        elif "train" in self.image_set:
            if self.transform:
                img = self.transform(img)
            if self.transforms:
                res = self.transforms(img, depth)
                return res
            if self.return_masks:
                return img, depth
            return img
        
    def __len__(self):
        return len(self.RGB_paths)



class NYUDataModule():
    """ 
    DataModule for Pascal VOC dataset

    Args:
        batch_size (int): batch size
        train_transform (torchvision.transforms): transform for training set
        val_transform (torchvision.transforms): transform for validation set
        test_transform (torchvision.transforms): transform for test set
        dir (str): path to dataset
        year (str): year of dataset
        split (str): split of dataset
        num_workers (int): number of workers for dataloader

    """
    def __init__(self,
                 data_dir: str,
                 train_transforms: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 train_split: str="train",
                 val_split: str="val",
                 val_image_transform: Optional[Callable]=None,
                 val_target_transform: Optional[Callable]=None,
                 val_transforms: Optional[Callable]=None,
                 shuffle: bool = True,
                 return_masks: bool = False,
                 drop_last: bool = False) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.return_masks = return_masks
        self.train_transforms = train_transforms
        self.val_image_transform = val_image_transform
        self.val_target_transform=val_target_transform
        self.val_transforms=val_transforms

    def setup(self):
        self.train_dataset = NYUDataset(self.data_dir, self.train_split, transforms=self.train_transforms, return_masks=self.return_masks)
        self.val_dataset = NYUDataset(self.data_dir, self.val_split, transform=self.val_image_transform, target_transform=self.val_target_transform, transforms=self.val_transforms, return_masks=True)
        # self.test_dataset = NYUDataset(self.dir, self.val_split, transform=self.val_image_transform, target_transform=self.val_target_transform, transforms=self.val_transforms, return_masks=True)

    def train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    # def get_test_dataloader(self, batch_size=None):
    #     batch_size = self.batch_size if batch_size is None else batch_size
    #     return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    # def get_test_dataset_size(self):
    #     return len(self.test_dataset)

    def get_module_name(self):
        return "NYUDataModule"
    
    def get_num_classes(self):
        return -1