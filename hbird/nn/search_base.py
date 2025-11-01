from abc import ABC, abstractmethod

class NearestNeighborSearchBase(ABC):
    """
    Abstract base class for nearest neighbor search.
    This enforces a consistent interface for Faiss-GPU and ScaNN implementations.
    """

    def __init__(self, feature_memory, n_neighbors=30, distance_measure="dot_product", **kwargs):
        self.feature_memory = feature_memory
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure.lower()
        self.device = feature_memory.device  # Determine if it's on CPU or GPU

        self.index = self._initialize_index()
        self._add_features_to_index()

    @abstractmethod
    def _initialize_index(self):
        """Initializes the nearest neighbor search index."""
        pass

    @abstractmethod
    def _add_features_to_index(self):
        """Adds feature vectors to the index."""
        pass

    @abstractmethod
    def find_nearest_neighbors(self, q, k=None):
        """Finds the nearest neighbors for a given query tensor."""
        pass
