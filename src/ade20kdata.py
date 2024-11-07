import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from typing import Optional
from PIL import Image

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
