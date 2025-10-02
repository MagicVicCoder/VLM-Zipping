from abc import ABC, abstractmethod
import torch

class BaseMLLM(ABC):
    """
    Abstract base class for Multi-Modal Large Language Models (MLLMs).
    Provides a unified interface for different MLLM architectures.
    """
    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Load the MLLM model and processor from Hugging Face or a local path.
        This method should populate self.model and self.processor.
        """
        pass

    @abstractmethod
    def get_components_for_env(self, image, question):
        """
        Process an image and question to get the components needed by the RL environment.
        
        This is a critical method that abstracts away the model-specific details of
        how visual and textual features are extracted and combined.

        Returns:
            A dictionary containing all necessary components, e.g.:
            {
                "original_visual_features": torch.Tensor,
                "text_embeds_part1": torch.Tensor,
                "text_embeds_part2": torch.Tensor,
                "query_embeddings": torch.Tensor,
                "current_num_patches": int
            }
        """
        pass
    
    @abstractmethod
    def generate_answer(self, final_embeddings, attention_mask, max_new_tokens=20):
        """
        Generate an answer given the final (pruned) embeddings.

        Args:
            final_embeddings (torch.Tensor): The combined text and visual embeddings after pruning.
            attention_mask (torch.Tensor): The corresponding attention mask.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated answer text.
        """
        pass

    @property
    def feature_dim(self):
        """
        Return the hidden size of the model's text config, which is used as the
        feature dimension for the observation space.
        """
        if self.model:
            return self.model.config.text_config.hidden_size
        return None
