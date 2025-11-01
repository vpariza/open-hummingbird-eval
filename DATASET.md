Please download the data and organize them in the folders are indicated below.
You can follow the section from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md)
to download and setup the datasets. We expect the following dataset folder structure.

## COCO
The structure for training should be as follows:
```
dataset root.
└───images
│   └─── train2017
│       │   *.jpg
│       │   ...
```

The structure for evaluating on COCO-Stuff and COCO-Things should be as follows:
```
dataset root.
└───annotations
│   └─── annotations
│       └─── stuff_annotations
│           │   stuff_val2017.json
│           └─── stuff_train2017_pixelmaps
│               │   *.png
│               │   ...
│           └─── stuff_val2017_pixelmaps
│               │   *.png
│               │   ...
│   └─── panoptic_annotations
│       │   panoptic_val2017.json
│       └─── semantic_segmenations_train2017
│           │   *.png
│           │   ...
│       └─── semantic_segmenations_val2017
│           │   *.png
│           │   ...
└───coco
│   └─── images
│       └─── train2017
│           │   *.jpg
│           │   ...
│       └─── val2017
│           │   *.jpg
│           │   ...
```
### TAR Structure
**Note**: In case you use a .tar file we expect the .tar file to explicitly have the `annotations` and `images` from above within it, without any root folder. 

## VOC Pascal
We provide you with a zipped version of the whole dataset as well as with two smaller versions of it:
* [Pascal VOC](https://1drv.ms/u/s!AnBBK4_o1T9MbXrxhV7BpGdS8tk?e=P7G6F0)
* [Mini Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EXkWjXPBLmNIgqI1G8yZzBYB_11wyXI-_8u0pyERgib8fA?e=qle36E)
* [Tiny Pascal VOC](https://1drv.ms/u/c/67fac29a77adbae6/EbGBdN6Z9LNEt3-3FveU344BnlECl_cwueg8-getyattqA?e=HPrVa1)
The structure for training and evaluation should be as follows:
```
dataset root.
├───VOCSegmentation
│   ├───SegmentationClass
│   │   │   *.png
│   │   │   ...
│   ├───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   │   *.png
│   │   │   ...
│   ├───images
│   │   │   *.jpg
│   │   │   ...
│   ├───sets
│   │   │   train.txt
│   │   │   trainaug.txt
│   │   │   val.txt
```
### TAR Structure
**Note**: In case you use a .tar file we expect the .tar file to explicitly have the `VOCSegmentation` folder with everything as the structure above.

## ADE20k
You can download the ADE20K dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset).
The structure for training and evaluation should be as follows:
```
dataset root.
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   ├── validation
│   ├── images
│   │   ├── training
│   │   ├── validation
```


## Cityscapes
The structure for training and evaluation should be as follows:
```
dataset root.
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   |   ├── aachen
│   │   |   |       aachen_000000_000019_gtFine_color.png
│   │   |   |       aachen_000000_000019_gtFine_instanceIds.png
│   │   |   |       aachen_000000_000019_gtFine_labelIds.png
│   │   |   |       ...
│   │   |   ├── ...
│   │   ├── val
│   │   |   ├── frankfurt
│   │   |   |       frankfurt_000000_000294_gtFine_color.png
│   │   |   |       frankfurt_000000_000294_gtFine_instanceIds.png
│   │   |   |       frankfurt_000000_000294_gtFine_labelIds.png
│   │   |   |       ...
│   │   |   ├── ...
│   │   ├── test
│   │   |   ├── berlin
│   │   |   |       berlin_000000_000019_gtFine_color.png
│   │   |   |       berlin_000000_000019_gtFine_instanceIds.png
│   │   |   |       berlin_000000_000019_gtFine_labelIds.png
│   │   |   |       ...
│   │   |   ├── ...
│   ├── leftImg8bit
│   │   ├── train
│   │   |   ├── aachen
│   │   |   |       aachen_000000_000019_leftImg8bit.png
│   │   |   |       ...
│   │   |   ├── ...
│   │   ├── val
│   │   |   ├── frankfurt
│   │   |   |       frankfurt_000000_000294_leftImg8bit.png
│   │   |   |       ...
│   │   |   ├── ...
│   │   ├── test
│   │   |   ├── berlin
│   │   |   |       berlin_000000_000019_leftImg8bit.png
│   │   |   |       ...
│   │   |   ├── ...
```
### TAR Structure
**Note**: In case you use a .tar file we expect the .tar file to explicitly have the `cityscapes` folder with everything as the structure above.