import torch
import faiss

from hbird.nn.search_base import NearestNeighborSearchBase

class NearestNeighborSearchFaiss(NearestNeighborSearchBase):
    def __init__(self, feature_memory, n_neighbors=30, distance_measure="dot_product", idx_shard=False, use_fp16=False, gpu_ids=None, **kwargs):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure.lower()
        self.idx_shard = idx_shard
        self.use_fp16=use_fp16
        self.embed_d = feature_memory.size(1)

        self.n_gpus = faiss.get_num_gpus()  # Get available GPUs
        if self.n_gpus < 1:
            raise RuntimeError("No GPUs available for Faiss.")

        # Set GPU IDs to use
        if gpu_ids is None:
            gpu_ids = list(range(self.n_gpus ))
        else:
            # Validate GPU IDs
            for gpu_id in gpu_ids:
                if gpu_id >= self.n_gpus  or gpu_id < 0:
                    raise ValueError(f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{self.n_gpus -1}")
        self.gpu_ids = gpu_ids

        # Initialize the Faiss index
        self.index = self._initialize_index()

        # Add feature vectors to the index
        self._add_features_to_index(feature_memory)

    def _get_sub_index(self, gpu_id, d:int):
        # Create resources for this GPU
        res = faiss.StandardGpuResources()
        
        # Create options for this GPU index
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = self.use_fp16  # Use FP16 for better performance
        config.device = gpu_id    # Specify which GPU to use

        if self.distance_measure == "dot_product":
            return faiss.GpuIndexFlatIP(res, d, config)  # Inner Product (Dot Product) index
        elif self.distance_measure in ["l2", "euclidean"]:
            return faiss.GpuIndexFlatL2(res, d, config)  # L2 (Euclidean) distance index
        else:
            raise ValueError(f"Unsupported distance measure: {self.distance_measure}")        

    def _initialize_index(self):
        d = self.embed_d

        if self.idx_shard:
            print("Using shard index")
            # Create shard index to distribute across GPUs
            gpu_index = faiss.IndexShards(d) 
            gpu_index.threaded = True
            # Note: threaded=True means search will be done in parallel threads
            
            # For each GPU, create a GPU index and add it to the shards
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu_sub_idx = self._get_sub_index(gpu_id, d)
                gpu_index.add_shard(gpu_sub_idx)

        else:
            print("Using Replicated index")
            gpu_index = faiss.IndexReplicas()
            
            # For each GPU, create a GPU index and add it to the replicas
            for i, gpu_id in enumerate(self.gpu_ids):
                # Create resources for this GPU
                res = faiss.StandardGpuResources()
                gpu_sub_idx = self._get_sub_index(gpu_id, d)
                gpu_index.addIndex(gpu_sub_idx)

        return gpu_index

    def _add_features_to_index(self, feature_memory):
        # Ensure feature_memory is on the CPU before adding to the index
        feature_memory_cpu = feature_memory.cpu().numpy()
        self.index.add(feature_memory_cpu)

    def find_nearest_neighbors(self, q, k=None):
        if k is None:
            k = self.n_neighbors

        # Ensure query tensor is on the CPU
        q_cpu = q.cpu().numpy()
        distances, indices = self.index.search(q_cpu, k)
        return indices, distances
