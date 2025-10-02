import random
from datasets import load_dataset
from .base_loader import BaseDataLoader

class VQADataLoader(BaseDataLoader):
    """
    Data loader for VQA datasets like MME.
    """
    def _load_and_split_data(self):
        print(f"Loading dataset: {self.name}")
        try:
            dataset = load_dataset(self.name, split=self.split)
        except Exception as e:
            print(f"Failed to load dataset {self.name}. Error: {e}")
            raise
        
        all_samples = [item for item in dataset]
        random.shuffle(all_samples)

        split_idx = int(self.split_ratio * len(all_samples))
        self.train_samples = all_samples[:split_idx]
        self.test_samples = all_samples[split_idx:]
        
        print(f"Dataset loaded and split: {len(self.train_samples)} for training, {len(self.test_samples)} for testing.")

def get_data_loader(config):
    """
    Factory function to get the specified data loader.
    """
    if config.DATASET_NAME == "lmms-lab/MME":
        return VQADataLoader(
            name=config.DATASET_NAME,
            split=config.DATASET_SPLIT,
            split_ratio=config.TRAIN_TEST_SPLIT_RATIO
        )
    # Add other datasets here
    # elif config.DATASET_NAME == "another-dataset":
    #     return AnotherDataLoader(...)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")
