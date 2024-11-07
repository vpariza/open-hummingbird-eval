if __name__ == "__main__":
    # Add project root to path if running this file as a main
    import sys
    import pathlib
    p = str(pathlib.Path(__file__).parent.resolve()) + '/'
    sys.path.append(p)


import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(iterator, *args, **kwargs):
        return iterator

import scann
import os

import numpy as np
from src.models import FeatureExtractorBeta as FeatureExtractor
from src.models import FeatureExtractorSimple
from src.eval_metrics import PredsmIoU

from src.transforms import get_hbird_val_transforms, get_hbird_train_transforms, get_hbird_train_transforms_for_imgs

from src.image_transformations import CombTransforms
from src.voc_data import VOCDataModule
from src.ade20kdata import Ade20kDataModule

class HbirdEvaluation():
    def __init__(self, feature_extractor, train_loader, n_neighbours, augmentation_epoch, num_classes, device, nn_params=None, memory_size=None, dataset_size=None, f_mem_p=None, l_mem_p=None):
        if nn_params is None:
            nn_params = {}
        self.feature_extractor = feature_extractor
        self.device = device
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.n_neighbours = n_neighbours
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.num_classes = num_classes
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.num_sampled_features = None
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p

        if self.memory_size is not None:
            self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch)
            ## create memory of specific size
            self.feature_memory = torch.zeros((self.memory_size, self.feature_extractor.d_model))
            self.label_memory = torch.zeros((self.memory_size, self.num_classes ))
        self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution)
        self.save_memory()
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        self.create_NN(self.n_neighbours, **nn_params)
    
    def create_NN(self, n_neighbours=30, distance_measure="dot_product", num_leaves=512, num_leaves_to_search=32, anisotropic_quantization_threshold=0.2, num_reordering_candidates=120, dimensions_per_block=4):
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), n_neighbours, distance_measure).tree(
    num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=anisotropic_quantization_threshold, dimensions_per_block=dimensions_per_block).reorder(num_reordering_candidates).build()

    def create_memory(self, train_loader, num_classes, eval_spatial_resolution):
        feature_memory = list()
        label_memory = list()
        idx = 0
        with torch.no_grad():
            for j in tqdm(range(self.augmentation_epoch), desc='Augmentation loop'):
                for i, (x, y) in enumerate(tqdm(train_loader, desc='Memory Creation loop')):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y = (y * 255).long()
                    y[y == 255] = 0
                    features, _ = self.feature_extractor.forward_features(x) # features of shape (BS, PS, D)
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    patchified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)
                    if self.memory_size is None:
                        # Memory Size is unbounded so we store all the features
                        normalized_features = features / torch.norm(features, dim=2, keepdim=True)

                        normalized_features = normalized_features.flatten(0, 1)
                        label = label.flatten(0, 2)
                        feature_memory.append(normalized_features.detach().cpu())
                        label_memory.append(label.detach().cpu())
                    else:
                        # Memory Size is bounded so we need to select/sample some features only
                        sampled_features, sampled_indices = self.sample_features(features, patchified_gts)
                        normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=2, keepdim=True)
                        label = label.flatten(1, 2)
                        ## select the labels of the sampled features
                        sampled_indices = sampled_indices.to(self.device)
                        ## repeat the label for each sampled feature
                        label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                        # label_hat = label.gather(1, sampled_indices)
                        normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                        label_hat = label_hat.flatten(0, 1)
                        self.feature_memory[idx:idx+normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                        self.label_memory[idx:idx+label_hat.size(0)] = label_hat.detach().cpu()
                        idx += normalized_sampled_features.size(0)
                        # memory.append(normalized_sampled_features.detach().cpu())
            if self.memory_size is None:
                self.feature_memory = torch.cat(feature_memory)
                self.label_memory = torch.cat(label_memory)

    def save_memory(self):
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)
    def load_memory(self):
        if self.f_mem_p is not None and self.l_mem_p is not None and os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p).to(self.device)
            self.label_memory = torch.load(self.l_mem_p).to(self.device)
            return True
        return False
    def sample_features(self, features, pathified_gts):
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(tqdm(pathified_gts)):
            class_frequency = self.get_class_frequency(gt)
            patch_scores, nonzero_indices = self.get_patch_scores(gt, class_frequency)

            patch_scores = patch_scores.flatten()
            nonzero_indices = nonzero_indices.flatten()

            # assert zero_score_idx[0].size(0) != 0 ## for pascal every patch should belong to one class
            patch_scores[~nonzero_indices] = 1e6

            # sample uniform distribution with the same size as the
            # number of nonzero indices (we use the sum here as the
            # nonzero_indices matrix is a boolean mask)
            uniform_x = torch.rand(nonzero_indices.sum())
            patch_scores[nonzero_indices] *= uniform_x
            feature = features[k]

            ### select the least num_sampled_features score indices
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)

            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)

        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)

        return sampled_features, sampled_indices

    def get_class_frequency(self, gt):
        class_frequency = torch.zeros((self.num_classes), device=self.device)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                class_frequency[patch_classes] += 1

        return class_frequency

    def get_patch_scores(self, gt, class_frequency):
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        nonzero_indices = torch.zeros((gt.shape[0], gt.shape[1]), dtype=torch.bool)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                patch_scores[i, j] = class_frequency[patch_classes].sum()
                nonzero_indices[i, j] = patch_classes.shape[0] > 0

        return patch_scores, nonzero_indices

    def patchify_gt(self, gt, patch_size):
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h//patch_size, patch_size, w//patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h//patch_size, w//patch_size, c*patch_size*patch_size)
        return gt

    def cross_attention(self, q, k, v, beta=0.02):
        """
        Args: 
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)
        """
        d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2) ## (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat
    
    def find_nearest_key_to_query(self, q):
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs*num_patches, d_k)
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).to(self.device)
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.n_neighbours, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.n_neighbours, -1)
        return key_features, key_labels

    def evaluate(self, val_loader, eval_spatial_resolution, return_knn_details=False, ignore_index=255):
        metric = PredsmIoU(self.num_classes, self.num_classes)
        self.feature_extractor = self.feature_extractor.to(self.device)
        label_hats = []
        lables = []

        knns = []
        knns_labels = []
        knns_ca_labels = []
        idx = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader, desc='Evaluation loop')):
                x = x.to(self.device)
                _, _, h, w = x.shape
                features, _ = self.feature_extractor.forward_features(x.to(self.device))
                features = features.to(self.device)
                y = y.to(self.device)
                y = (y * 255).long()
                ## copy the data of features to another variable
                q = features.clone()
                q = q.detach().cpu().numpy()
                key_features, key_labels = self.find_nearest_key_to_query(q)           
                label_hat =  self.cross_attention(features, key_features, key_labels)
                if return_knn_details:
                    knns.append(key_features.detach().cpu())
                    knns_labels.append(key_labels.detach().cpu())
                    knns_ca_labels.append(label_hat.detach().cpu())
                bs, _, label_dim = label_hat.shape
                label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
                resized_label_hats =  F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)
                label_hats.append(cluster_map.detach().cpu())
                lables.append(y.detach().cpu())

            lables = torch.cat(lables) 
            label_hats = torch.cat(label_hats)
            valid_idx = lables != ignore_index
            valid_target = lables[valid_idx]
            valid_cluster_maps = label_hats[valid_idx]
            metric.update(valid_target, valid_cluster_maps)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            if return_knn_details:
                knns = torch.cat(knns)
                knns_labels = torch.cat(knns_labels)
                knns_ca_labels = torch.cat(knns_ca_labels)
                return jac, {"knns": knns, "knns_labels": knns_labels, "knns_ca_labels": knns_ca_labels}
            else:
                return jac


def hbird_evaluation(model, d_model, patch_size, dataset_name:str, data_dir:str, batch_size=64, input_size=224, 
                        augmentation_epoch=1, device='cpu', return_knn_details=False, n_neighbours=30, nn_params=None, 
                        ftr_extr_fn=None, memory_size=None, num_workers=8, ignore_index=255):
    eval_spatial_resolution = input_size // patch_size
    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    else:
        feature_extractor = FeatureExtractorSimple(model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    train_transforms_dict = get_hbird_train_transforms(input_size)
    val_transforms_dict = get_hbird_val_transforms(input_size)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
    
    dataset_size = 0
    num_classes = 0
    ignore_index = -1   
    if dataset_name == "voc":
        ignore_index = 255
        dataset = VOCDataModule(batch_size=batch_size,
                                    num_workers=num_workers,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True)
        dataset.setup()
    elif dataset_name == "ade20k":
        ignore_index = 0
        dataset = Ade20kDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size)
        dataset.setup()
    else:
        raise ValueError("Unknown dataset name")

    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    evaluator = HbirdEvaluation(feature_extractor, train_loader, n_neighbours=n_neighbours, 
                        augmentation_epoch=augmentation_epoch, num_classes=num_classes, 
                        device=device, nn_params=nn_params, memory_size=memory_size, 
                        dataset_size=dataset_size)
    return evaluator.evaluate(val_loader, eval_spatial_resolution, return_knn_details=return_knn_details)
           
    