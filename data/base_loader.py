from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    All data loaders should inherit from this class and implement its methods.
    """
    def __init__(self, name, split, split_ratio):
        self.name = name
        self.split = split
        self.split_ratio = split_ratio
        self.train_samples = []
        self.test_samples = []
        self._load_and_split_data()

    @abstractmethod
    def _load_and_split_data(self):
        """
        Load the dataset and split it into training and testing sets.
        This method should populate self.train_samples and self.test_samples.
        """
        pass

    def get_train_samples(self):
        return self.train_samples

    def get_test_samples(self):
        return self.test_samples
