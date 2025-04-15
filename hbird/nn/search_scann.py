import scann
import numpy as np

from hbird.nn.search_base import NearestNeighborSearchBase

class NearestNeighborSearchScaNN(NearestNeighborSearchBase):
    def __init__(self, feature_memory, n_neighbors=30, distance_measure="dot_product",
                 num_leaves=512, num_leaves_to_search=32,
                 anisotropic_quantization_threshold=0.2, num_reordering_candidates=120,
                 dimensions_per_block=4, **kwargs):
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.num_reordering_candidates = num_reordering_candidates
        self.dimensions_per_block = dimensions_per_block
        super().__init__(feature_memory, n_neighbors, distance_measure)

    def _initialize_index(self):
        if self.distance_measure not in ["dot_product", "euclidean"]:
            raise ValueError(f"Unsupported distance measure: {self.distance_measure}")

        index = scann.scann_ops_pybind.builder(self.feature_memory.cpu().numpy(),
                                               self.n_neighbors,
                                               self.distance_measure)

        index = index.tree(num_leaves=self.num_leaves, num_leaves_to_search=self.num_leaves_to_search,
                           training_sample_size=self.feature_memory.size(0))

        index = index.score_ah(2, anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
                               dimensions_per_block=self.dimensions_per_block)

        index = index.reorder(self.num_reordering_candidates)
        return index.build()

    def _add_features_to_index(self):
        """ScaNN does not need an explicit method to add features, as it builds the index directly."""
        pass

    def find_nearest_neighbors(self, q, k=None):
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs * num_patches, d_k).cpu().numpy()
        neighbors, distances = self.index.search_batched(reshaped_q)
        return neighbors, distances
