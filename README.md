### Description
This repository implements the Dense NN Retrieval Evaluation method introduced by Balažević et al. [Towards In-context Scene Understanding](https://arxiv.org/abs/2306.01667).

Briefly, it evaluates the effectiveness of spatial features acquired from a vision encoder, to associate themselves to relevant features from a dataset (validation), through the utilization of a k-NN classifier/retriever that operates across various proportions of training data.

### Notes
* For any questions/issues etc. please open a github issue on this repository.
* If you find this repository useful, please consider starring

### Usage

#### Example on how to Evaluate dino with Dense NN Retrieval on Pascal VOC

```python
import torch
from src.dense_nn_ret_eval import dense_nn_ret_evaluation
# Parameters for the model dino
device = 'cuda'
input_size = 224
batch_size = 64
patch_size = 16
embed_dim = 384
vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# Define the function to extract features from the model
# Input to the function is the model and the images
# Output of the function is the features extracted from the model and optionally the attention maps
fn = lambda model, imgs: (model.forward_backbone(imgs)[:, 1:], None)

# Evaluate the model using the Full In-Context Learning Dense k-NN Retrieval Evaluation on the Pascal VOC Dataset
hbird_miou_score = dense_nn_ret_evaluation(model.to(device), 
                            d_model=embed_dim,          # size of the embedding feature vectors of patches
                            patch_size=patch_size, 
                            batch_size = batch_size, 
                            input_size=224,             
                            augmentation_epoch=1,       # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
                            device=device,              
                            return_knn_details=False,   # whether to return additional NNs details
                            num_neighbour=30,           # the number of neighbors to fetch per image patch
                            nn_params=None,             # Other parameters to be used for the k-NN operator
                            ftr_extr_fn=fn,             # function that extracts features from a vision encoder on images
                            dataset_name='voc',         # the name of the dataset to use, currently only Pascal VOC is included. But it is easy to add other ones
                            data_dir='<the path to the Pascal VOC Dataset>',    # path to the dataset to use for evaluation
                            memory_size=None)           # How much you want to limit your memory size, None if to be left unbounded
print('Dense NN Ret - miou score:', hbird_miou_score) 

```

#### Example on how to Evaluate dinov2 with Dense NN Retrieval on Pascal VOC

```python
import torch
from src.dense_nn_ret_eval import dense_nn_ret_evaluation
# Parameters for the model dino
device = 'cuda'
input_size = 224
batch_size = 256
patch_size = 14
embed_dim = 384
vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the function to extract features from the model
# Input to the function is the model and the images
# Output of the function is the features extracted from the model and optionally the attention maps
fn = lambda model, imgs: (model.forward_backbone(imgs)[:, 1:], None)

# Evaluate the model using the Full In-Context Learning Dense k-NN Retrieval Evaluation on the Pascal VOC Dataset
hbird_miou_score = dense_nn_ret_evaluation(model.to(device), 
                            d_model=embed_dim,          # size of the embedding feature vectors of patches
                            patch_size=patch_size, 
                            batch_size = batch_size, 
                            input_size=224,             
                            augmentation_epoch=1,       # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
                            device=device,              
                            return_knn_details=False,   # whether to return additional NNs details
                            num_neighbour=30,           # the number of neighbors to fetch per image patch
                            nn_params=None,             # Other parameters to be used for the k-NN operator
                            ftr_extr_fn=fn,             # function that extracts features from a vision encoder on images
                            dataset_name='voc',         # the name of the dataset to use, currently only Pascal VOC is included. But it is easy to add other ones
                            data_dir='<the path to the Pascal VOC Dataset>',    # path to the dataset to use for evaluation
                            memory_size=None)           # How much you want to limit your dataset, None if to be left unbounded
print('Dense NN Ret - miou score:', hbird_miou_score) 

```
###  Setup
This is the section describing what is required to execute the Dense NN Retrieval Evaluation.

#### Python Libraries
The most important librari
* `torch`
* `scann`
* `numpy`
* `joblib`

#### Dataset Setup
##### VOC Pascal
The structure of the Pascal VOC dataset folder should be as follows:
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

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

