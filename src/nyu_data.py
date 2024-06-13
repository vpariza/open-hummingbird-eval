import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Any
from pathlib import Path
from typing import Optional, Callable
from transforms import get_default_val_transforms

import torch
from abc import ABC, abstractmethod
import os
import pandas as pd

import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt


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
    ):
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
        
        file_paths = pd.read_csv(csv_path)
        self.RGB_paths = [os.path.join(self.root, img) for img in file_paths.iloc[:, 0]]
        self.depth_paths = [os.path.join(self.root, img) for img in file_paths.iloc[:, 1]]

        assert len(self.RGB_paths) == len(self.depth_paths) 



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        print(self.RGB_paths[index])
        print(self.depth_paths[index])
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

    def __init__(self, batch_size, train_transform, val_transform, test_transform,  dir, num_workers=0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dir = dir
        self.image_train_transform = train_transform["img"]
        self.image_val_transform = val_transform["img"]
        self.image_test_transform = test_transform["img"]
        self.target_train_transform = None
        self.target_val_transform = None
        self.target_test_transform = None
        self.shared_train_transform = train_transform["shared"]
        self.shared_val_transform = val_transform["shared"]
        self.shared_test_transform = test_transform["shared"]

    def setup(self):
        self.train_dataset = NYUDataset(self.dir, "train", transform=self.image_train_transform, target_transform=self.target_train_transform, transforms=self.shared_train_transform)
        self.val_dataset = NYUDataset(self.dir, "val", transform=self.image_test_transform, target_transform=self.target_test_transform, transforms=self.shared_test_transform)
        self.test_dataset = NYUDataset(self.dir, "val", transform=self.image_test_transform, target_transform=self.target_test_transform, transforms=self.shared_test_transform)

    def get_train_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def get_val_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_test_dataloader(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    
    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)

    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def get_module_name(self):
        return "NYUDataModule"
    
    def get_num_classes(self):
        return -1


if __name__ == "__main__":
    input_size = 224
    train_transforms = get_default_val_transforms(input_size)
    val_transforms = get_default_val_transforms(input_size)
    test_transforms = get_default_val_transforms(input_size)
    dataset_size = 0
    num_classes = 0
    batch_size = 1
    data_dir = "/ssdstore/ssalehi/nyu_data"
    dataset = NYUDataModule(batch_size=batch_size, train_transform=train_transforms, val_transform=val_transforms, test_transform=val_transforms, dir=data_dir)
    dataset.setup()
    dataset_size = dataset.get_train_dataset_size()
    num_classes=dataset.get_num_classes()
    train_loader = dataset.get_train_dataloader()
    val_loader = dataset.get_val_dataloader()

    print(f"Dataset Size: {dataset_size}")
    print(f"Number of Classes: {num_classes}")
    for i, (rgb, depth) in enumerate(train_loader):
        print(rgb.shape)
        print(depth.shape)
        print(rgb[0].max())
        print(depth[0].max())
        if i == 3:
            break
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(rgb[0].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(depth[0].permute(1, 2, 0))
        plt.show()
        plt.savefig(f"output_{i}.png")
        
