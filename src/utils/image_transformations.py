import random
from PIL import Image
# Custom function to apply resizing and cropping with probability
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import numpy as np
from PIL import ImageFilter

def random_resize_crop(image, target, size=(256, 256), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    image = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
    target = F.resized_crop(target, i, j, h, w, size, interpolation=Image.NEAREST)
    return image, target

def resize(image, target, size=(256, 256)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    image = F.resize(image, size, interpolation=Image.BILINEAR)
    target = F.resize(target, size, interpolation=Image.NEAREST)
    return image, target


def apply_horizontal_flip(image, target):
    # Generate a random seed for the transformation
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    seed = torch.randint(0, 2**32, size=(1,)).item()
    torch.manual_seed(seed)

    # Apply horizontal flip to the image
    image = F.hflip(image)

    # Use the same seed for the target to ensure consistent flip
    torch.manual_seed(seed)
    target = F.hflip(target)

    return image, target


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.5, 2), ratio=(3. / 4., 4. / 3.), probability=1.0):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return random_resize_crop(img, target, self.size, self.scale, self.ratio)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return apply_horizontal_flip(img, target)
        return img, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, target):
        return resize(img, target, self.size)

class TwoViewTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        return self.transforms(img), self.transforms(img)

class MultiCropTransform(object):
    def __init__(self, 
            custom_trans:list=None, 
            mean=[0.485, 0.456, 0.406],
            std=[0.228, 0.224, 0.225],
            size_crops=[224, 224],
            nmb_crops=[2, 6],
            min_scale_crops=[0.14, 0.05],
            max_scale_crops=[1., 0.14],
            local_view_trans=None,
            n_views_per_crop=1):
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            if custom_trans is None:
                if n_views_per_crop == 1:
                    trans.extend([transforms.Compose([
                        randomresizedcrop,
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Compose(color_transform),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
                    ] * nmb_crops[i])
                elif n_views_per_crop == 2:
                    if local_view_trans is None:
                        local_view_trans = transforms.Compose([transforms.Compose(color_transform),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
                    trans.extend([transforms.Compose([
                        randomresizedcrop,
                        transforms.RandomHorizontalFlip(p=0.5),
                        TwoViewTransform(local_view_trans)
                        ])
                    ] * nmb_crops[i])
                else :
                    raise ValueError("Only 1 or 2 views are supported")
            else:
                trans.extend([custom_trans] * nmb_crops[i])
        self.trans = trans
        self
    
    def __call__(self, img):
        return list(map(lambda trans: trans(img), self.trans))


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class SepTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        return img, tgt

class ToFloat(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.float()


class CombTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None, img_tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform
        self.img_tgt_transform = img_tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        if self.img_tgt_transform:
            return self.img_tgt_transform(img, tgt)
        else:
            return img, tgt