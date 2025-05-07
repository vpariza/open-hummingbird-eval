Please download the data and organize them in the folders are indicated below.
You can follow the section from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md)
to download and setup the datasets. We expect the following dataset folder structure.

##### COCO
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
##### VOC Pascal
We provide you with a zipped version of the whole dataset as well as with two smaller versions of it:
* [Pascal VOC](https://drive.google.com/uc?id=1LITii48SINBFAankzh_HlYWx5jCJzG4X)
* [Mini Pascal VOC](https://drive.google.com/uc?id=1GoRApuZzsM8a5nTa-wyEsLPOXPJuzbcT)
* [Tiny Pascal VOC](https://drive.google.com/uc?id=1o9evnop5BJnT47jwFk9cUp3TxHRdNpVN)
The structure for training and evaluation should be as follows:
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```

##### ADE20k
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


##### Cityscapes
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