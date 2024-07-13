if __name__ == "__main__":
    # Add project root to path if running this file as a main
    import sys
    import pathlib
    p = str(pathlib.Path(__file__).parent.resolve()) + '/'
    sys.path.append(p)


# Add project root to path if running this file as a main
import sys
import pathlib
p = str(pathlib.Path('.').parent.resolve()) + '/'
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
from src.utils.eval_models import FeatureExtractorBeta as FeatureExtractor
from src.utils.eval_models import FeatureExtractorSimple
from src.utils.eval_metrics import PredsmIoU

from src.utils.transforms import get_hbird_val_transforms, get_hbird_train_transforms, get_hbird_train_transforms_for_imgs

from src.utils.image_transformations import CombTransforms
from data.VOCdevkit.vocdata_2 import VOCDataModule
from data.ade20k.ade20kdata import Ade20kDataModule
from data.nyu.nyu_data import NYUDataModule

from src.utils.transforms import get_default_val_transforms
# Version 2

def rmse(x,y, eps = 1e-6):
    return torch.sqrt(torch.nn.functional.mse_loss(x, y) + eps)

class HbirdEvaluation():
    def __init__(self, feature_extractor, train_loader, n_neighbours, augmentation_epoch, num_classes, device, sampling_fract_num=None,nn_params=None, memory_size=None, 
                        dataset_size=None, f_mem_p=None, l_mem_p=None, sampled_indices_p=None, sampling_approach=None, is_target_int=False):
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

        self.is_target_int = is_target_int
        self.sampled_indices_p = sampled_indices_p
        self.sampling_fract_num = sampling_fract_num if sampling_fract_num is not None else 1

        self.sampled_indices_all = None  # keeps track of the sampled indices of patches per image of the augmented datasets for reproducibility purposes

        assert sampling_approach=="random" or sampling_approach=="u_patch_frequency" or sampling_approach is None # uniformly randomly or the unique patch frequency approach 
        
        self.sampling_patch_fn = None
        if sampling_approach == "u_patch_frequency":
            self.sampling_patch_fn = self.sample_features
        elif sampling_approach == "random":
            self.sampling_patch_fn = self.sample_features_randomly

        if self.memory_size is not None:
            # define how many sampled patches to get per image
            # print('self.memory_size',self.memory_size)
            # print('dataset_size',dataset_size)
            # print('self.augmentation_epoch',self.augmentation_epoch)
            self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch)
            # create memory of specific size for features and labels
            self.feature_memory = torch.zeros((self.memory_size, self.feature_extractor.d_model))
            self.label_memory = torch.zeros((self.memory_size, self.num_classes if self.num_classes > 0 else 1))
        if self.load_memory() == False:
            # if memory was not loaded then try to load the indices of the patches selected from each augmented image
            loaded_sample_idces = self.load_sampled_indices()
            self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution, sel_sampled_indices=self.sampled_indices_all)
            if loaded_sample_idces == False:
                # save the sampled indices if they have not already been loaded
                self.save_sampled_indices()
            self.save_memory()
        # Move memory of features and labels to the specified device
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)

        self.create_NN(self.n_neighbours, **nn_params)
    
    def create_NN(self, n_neighbours=30, distance_measure="dot_product", num_leaves=512, num_leaves_to_search=32, anisotropic_quantization_threshold=0.2, num_reordering_candidates=120):
    # def create_NN(self, n_neighbours=30, distance_measure="dot_product", num_leaves=128, num_leaves_to_search=32, anisotropic_quantization_threshold=0.2, num_reordering_candidates=32):
        self.NN_algorithm = scann.scann_ops_pybind.builder(self.feature_memory.detach().cpu().numpy(), n_neighbours, distance_measure).tree(
    num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=self.feature_memory.size(0)).score_ah(
    2, anisotropic_quantization_threshold=anisotropic_quantization_threshold).reorder(num_reordering_candidates).build()

    def create_memory(self, train_loader, num_classes, eval_spatial_resolution, sel_sampled_indices=None):
        feature_memory = list()
        label_memory = list()
        idx = 0
        sampled_indices_all = list()
        bs_idx = 0
        with torch.no_grad():
            for j in tqdm(range(self.augmentation_epoch), desc='Augmentation loop'):
                for i, (x, y) in enumerate(tqdm(train_loader, desc='Memory Creation loop')):
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    # Step 1: Load Images and Target Labels
                    bs = x.shape[0]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # Step 2: Preprocess the target labels
                    ## Transform floating point labels between [0,1] to integers between [0,255]
                    if self.is_target_int == True:
                        y = (y * 255).long()
                        ## ignore the label of 255
                        y[y == 255] = 0
                    # Step 3: Encode images
                    features, _ = self.feature_extractor.forward_features(x)
                    # Step 4: Patchfify target labels to have the same shape as the features
                    patchified_gts = self.patchify_gt(y, patch_size) # of shape (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    if self.is_target_int == True:
                        # Convert integer labels to one hot encoding
                        patch_gts = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    else:
                        patch_gts = patchified_gts.unsqueeze(-1) 
                    label = patch_gts.mean(dim=3)
                    if self.memory_size is None or self.sampling_patch_fn is None:
                        # Memory Size is unbounded so we store all the features
                        normalized_features = features / torch.norm(features, dim=1, keepdim=True)

                        normalized_features = normalized_features.flatten(0, 1)
                        label = label.flatten(0, 2)
                        feature_memory.append(normalized_features.detach().cpu())
                        label_memory.append(label.detach().cpu())
                    else:
                        for i in tqdm(range(0, self.sampling_fract_num), desc='Sampling loop'):
                            if sel_sampled_indices is None:
                                # Memory Size is bounded so we need to select/sample some features only
                                # sampled_features, sampled_indices = self.sample_features(features, patchified_gts)
                                # sampled_features, sampled_indices = self.sample_features_randomly(features, patchified_gts)
                                sampled_features, sampled_indices = self.sampling_patch_fn(features, patchified_gts)
                                sampled_indices_all.append(sampled_indices.detach().cpu())
                            else:
                                sampled_indices = sel_sampled_indices[bs_idx*bs: (bs_idx+1)*bs]
                                sampled_features = torch.gather(features, 1, sampled_indices.unsqueeze(-1).repeat(1, 1, features.shape[-1]).to(self.device))

                            normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=1, keepdim=True)
                            # combine the dimensions of patches across h and w 
                            label_hat = label.flatten(1, 2)
                            ## select the labels of the sampled features
                            sampled_indices = sampled_indices.to(self.device)
                            # print('sampled_indices',sampled_indices.shape)
                            # print('label',label.shape)
                            # print('label_hat',label_hat.shape)
                            # print('sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1])',sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]).shape)
                            label_hat = label_hat.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                            normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                            label_hat = label_hat.flatten(0, 1)
                            self.feature_memory[idx:idx+normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                            self.label_memory[idx:idx+label_hat.size(0)] = label_hat.detach().cpu()
                            idx += normalized_sampled_features.size(0)
                            bs_idx += 1
            if self.memory_size is None or self.sampling_patch_fn is None:
                self.feature_memory = torch.cat(feature_memory)
                self.label_memory = torch.cat(label_memory)
            
            if sel_sampled_indices is None and self.memory_size is not None:
                self.sampled_indices_all = torch.cat(sampled_indices_all)
            else:
                self.sampled_indices_all = sel_sampled_indices

    def save_memory(self):
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)

    def load_memory(self):
        if self.f_mem_p is not None and self.l_mem_p is not None and os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p).to(self.device)
            self.label_memory = torch.load(self.l_mem_p).to(self.device)
            print('Memory features loaded from', self.f_mem_p)
            print('Memory labels from', self.l_mem_p)
            return True
        return False

    def save_sampled_indices(self):
        if self.sampled_indices_p is not None and self.sampled_indices_all is not None:
            torch.save(self.sampled_indices_all.cpu(), self.sampled_indices_p)
            print('Sampled indices save to', self.sampled_indices_p)

    def load_sampled_indices(self):
        if self.sampled_indices_p is not None and os.path.isfile(self.sampled_indices_p):
            self.sampled_indices_all = torch.load(self.sampled_indices_p, map_location='cpu')
            print('Sampled indices loaded from', self.sampled_indices_p)
            return True
        else:
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
            ### select the least num_sampled_features score idndices
            # print('patch_scores',patch_scores.shape)
            # print('self.num_sampled_features',self.num_sampled_features)
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)
            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)
        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)
        return sampled_features, sampled_indices

    def sample_features_randomly(self, features, pathified_gts):
        bs, ps, d = features.shape
        sampled_indices = torch.argsort(torch.rand((bs, ps), device=self.device), dim=1)[:,:self.num_sampled_features]
        sampled_features = torch.gather(features, 1, sampled_indices.unsqueeze(-1).repeat(1,1,d))
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
                # Step 1: Load images to the specified device
                x = x.to(self.device)
                y = y.to(self.device)
                _, _, h, w = x.shape
                if self.is_target_int == True:
                    # Step 1-b: Transform floating point labels between [0,1] to integers between [0,255]
                    y = (y * 255).long()
                # Step 3: Encode images
                features, _ = self.feature_extractor.forward_features(x.to(self.device))
                features = features.to(self.device)
                ## copy the data of features to another variable
                q = features.clone()
                q = q.detach().cpu().numpy()
                # Step 4: Find k-NNs to each encoded patch
                key_features, key_labels = self.find_nearest_key_to_query(q)  
                # print('key_features',key_features.shape)  
                # print('key_labels',key_labels.shape)  
                # print('feature_memory',self.feature_memory.shape)
                # print('label_memory',self.label_memory.shape)
                # Step 5: Apply cross-attention on the k retrieved neighbors
                label_hat =  self.cross_attention(features, key_features, key_labels)
                ## 
                if return_knn_details:
                    knns.append(key_features.detach().cpu())
                    knns_labels.append(key_labels.detach().cpu())
                    knns_ca_labels.append(label_hat.detach().cpu())
                bs, _, label_dim = label_hat.shape
                label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
                resized_label_hats =  F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
                # print('resized_label_hats',resized_label_hats.shape)
                if self.is_target_int == True:
                    gt_map = resized_label_hats.argmax(dim=1).unsqueeze(1)
                else:
                    gt_map = resized_label_hats
                # print('gt_map',gt_map.shape)
                label_hats.append(gt_map.detach().cpu())
                lables.append(y.detach().cpu())
                idx += x.size(0)

            lables = torch.cat(lables) 
            label_hats = torch.cat(label_hats)
            # filter out labels that should not be checked
            if self.is_target_int == True:
                valid_idx = lables != ignore_index
                valid_target = lables[valid_idx]
                valid_cluster_maps = label_hats[valid_idx]
                # compute miou between ground truth labels and inferred labels
                metric.update(valid_target, valid_cluster_maps)
                res, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)
            else:
                res = rmse(lables, label_hats)
                
                
            if return_knn_details:
                knns = torch.cat(knns)
                knns_labels = torch.cat(knns_labels)
                knns_ca_labels = torch.cat(knns_ca_labels)
                return res, {"knns": knns, "knns_labels": knns_labels, "knns_ca_labels": knns_ca_labels}
            else:
                return res

def hbird_evaluation(model, d_model, patch_size, dataset_name:str, data_dir:str, batch_size=64, 
                    input_size=224, augmentation_epoch=1, device='cpu', return_knn_details=False, 
                    n_neighbours=30, nn_params=None, ftr_extr_fn=None, memory_size=None, num_workers=8, 
                    eval_dir=None, ignore_index=255, method:str=None, sampling_fract_num=None, seed=400):
    eval_spatial_resolution = input_size // patch_size
    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    else:
        feature_extractor = FeatureExtractorSimple(model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    train_transforms_dict = get_hbird_train_transforms(input_size, n_views=1)
    val_transforms_dict = get_hbird_val_transforms(input_size)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
    sampling_approach=None
    is_target_int=False
    dataset_size = 0
    num_classes = 0
    ignore_index = -1   
    if dataset_name == "voc":
        ignore_index = 255
        image_set = "trainaug"
        if sampling_fract_num is not None:
            image_set = f'{image_set}_{sampling_fract_num}_{seed}'
        dataset = VOCDataModule(batch_size=batch_size,
                                    num_workers=num_workers,
                                    train_split=image_set,
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True)
        dataset_size = 10582
        dataset.setup()
        sampling_approach = "u_patch_frequency"
        is_target_int = True
    elif dataset_name == "ade20k":
        ignore_index = 0
        file_set = None
        image_set = 'training'
        if image_set is not None and sampling_fract_num is not None:
            file_set_file = os.path.join("/mnt/beegfs/vpariz01/workspace/sets_ade20k", f'{image_set}_{sampling_fract_num}_{seed}.txt')
            with open(file_set_file, "r") as f:
                file_set = [x.strip() for x in f.readlines()]
        dataset = Ade20kDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size,
                 train_file_set=file_set)
        dataset_size = 20210
        dataset.setup()
        sampling_approach = "u_patch_frequency"
        is_target_int = True
    elif dataset_name == "nyuv2":
        train_transforms_dict = get_default_val_transforms(input_size)
        val_transforms_dict = get_default_val_transforms(input_size)
        train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
        val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
        dataset = NYUDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size)
        dataset.setup()
        dataset_size = dataset.get_train_dataset_size()
        # val_dataset_size = dataset.get_val_dataset_size()
        # num_classes=dataset.get_num_classes()
        sampling_approach = "random"
        is_target_int = False
    else:
        raise ValueError("Unknown dataset name")

    # dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    sampled_indices_p = None
    f_mem_p = None
    l_mem_p = None
    if eval_dir is not None:
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        # if sampling_fract_num is not None:
        #     sampled_indices_p = os.path.join(eval_dir, f'sampled_indices_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_{sampling_fract_num}.pth')
        #     f_mem_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_{sampling_fract_num}_fs.pth')
        #     l_mem_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_{sampling_fract_num}_ls.pth')
        # else:
        #     sampled_indices_p = os.path.join(eval_dir, f'sampled_indices_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}.pth')
        #     f_mem_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_fs.pth')
        #     l_mem_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_ls.pth')
        sampled_indices_p = os.path.join(eval_dir, f'sampled_indices_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}.pth')
        # val_l_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ps_{patch_size}_val_ls.pth')
        # val_f_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ps_{patch_size}_val_fs.pth')
        # val_l_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_val_ls.pth')
        # val_f_p = os.path.join(eval_dir, f'{method}_{dataset_name}_ms_{memory_size}_ps_{patch_size}_aug_{augmentation_epoch}_val_fs.pth')
    evaluator = HbirdEvaluation(feature_extractor, train_loader, n_neighbours=n_neighbours, 
            augmentation_epoch=augmentation_epoch, num_classes=num_classes, device=device, 
            nn_params=nn_params, memory_size=memory_size, dataset_size=dataset_size, 
            sampled_indices_p=sampled_indices_p, f_mem_p=f_mem_p, l_mem_p=l_mem_p, 
            sampling_fract_num=sampling_fract_num, sampling_approach=sampling_approach,
            is_target_int=is_target_int)
    miou = evaluator.evaluate(val_loader, eval_spatial_resolution, return_knn_details=return_knn_details, ignore_index=ignore_index)
    # print('hbird mIoU:', miou)
    return miou
           