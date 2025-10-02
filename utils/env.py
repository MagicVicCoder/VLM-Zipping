import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from PIL import Image

class MLLMTokenPruningEnv(gym.Env):
    """
    RL Environment for MLLM token pruning.
    This environment is designed to be modular and work with different MLLMs and datasets.
    """
    def __init__(self, mllm_wrapper, vqa_samples, config, seed=None):
        super().__init__()
        self.mllm = mllm_wrapper
        self.vqa_samples = vqa_samples
        self.config = config
        self.device = config.DEVICE
        self._env_seed = seed
        self._set_seed(seed)
        if not self.vqa_samples:
            raise ValueError("VQA samples list cannot be empty.")
        self.feature_dim = self.mllm.feature_dim
        self.max_num_patches = self.config.MAX_PATCHES
        # Episode-specific state
        self.decisions_made = 0
        self.pruning_mask = None
        self.current_question = ""
        self.gt_answer = ""
        self.current_num_patches = 0
        # MLLM-related components for the current episode
        self.original_visual_features = None
        self.initial_mean_features = None
        self.text_embeds_part1 = None
        self.text_embeds_part2 = None
        self.query_embeddings = None
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([self.max_num_patches, 2])
        self.observation_space = spaces.Dict({
            "visual_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_num_patches, self.feature_dim)),
            "query_embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.feature_dim)),
            "pruning_mask": spaces.Box(low=-1, high=1, shape=(self.max_num_patches,), dtype=np.int8),
            "valid_token_mask": spaces.Box(low=0, high=1, shape=(self.max_num_patches,), dtype=np.int8)
        })

    def _set_seed(self, seed):
        if seed is not None:
            import random, numpy as np, torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _get_vqa_sample(self):
        return random.choice(self.vqa_samples)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._set_seed(seed)
        super().reset(seed=seed)
        while True:
            sample = self._get_vqa_sample()
            image, question, answer = sample.get('image'), sample.get('question'), sample.get('answer')
            if not isinstance(image, Image.Image) or not question or not answer:
                continue
            components = self.mllm.get_components_for_env(image, question)
            if components:
                break
        self.current_question = question
        self.gt_answer = answer
        self.decisions_made = 0
        self.original_visual_features = components["original_visual_features"]
        self.current_num_patches = min(components["current_num_patches"], self.max_num_patches)
        if self.original_visual_features.shape[1] > self.max_num_patches:
            self.original_visual_features = self.original_visual_features[:, :self.max_num_patches, :]
        self.initial_mean_features = self.original_visual_features.mean(dim=1)
        self.text_embeds_part1 = components["text_embeds_part1"]
        self.text_embeds_part2 = components["text_embeds_part2"]
        self.query_embeddings = components["query_embeddings"]
        self.pruning_mask = np.full(self.max_num_patches, -1, dtype=np.int8)
        return self._get_obs(), {}

    def _get_obs(self):
        padded_features = np.zeros((self.max_num_patches, self.feature_dim), dtype=np.float32)
        if self.original_visual_features is not None:
             actual_features = self.original_visual_features.squeeze(0).cpu().numpy()
             padded_features[:self.current_num_patches, :] = actual_features

        valid_token_mask = np.zeros(self.max_num_patches, dtype=np.int8)
        valid_token_mask[:self.current_num_patches] = 1

        return {
            "visual_features": padded_features,
            "query_embeddings": self.query_embeddings.cpu().numpy() if self.query_embeddings is not None else np.zeros((1, self.feature_dim), dtype=np.float32),
            "pruning_mask": self.pruning_mask.copy(),
            "valid_token_mask": valid_token_mask
        }

    def step(self, action):
        index, decision = int(action[0]), int(action[1])
        
        if index >= self.current_num_patches or self.pruning_mask[index] != -1:
            reward = -2.0
            done = True
            info = {"error": "Invalid action"}
            return self._get_obs(), reward, done, False, info

        self.pruning_mask[index] = decision
        self.decisions_made += 1

        # if self.decisions_made % 1000 == 0 or self.decisions_made == self.current_num_patches:
        #     print(f"Decisions made: {self.decisions_made}/{self.current_num_patches}")
        
        done = self.decisions_made >= self.current_num_patches
        
        num_pruned = (self.pruning_mask[:self.current_num_patches] == 0).sum()
        r_efficiency = self.config.BETA * (num_pruned / self.current_num_patches) if self.current_num_patches > 0 else 0
        
        r_semantic = 0.0
        if not done:
            kept_indices = np.where(self.pruning_mask[:self.current_num_patches] == 1)[0]
            if len(kept_indices) > 0:
                with torch.no_grad():
                    kept_indices_tensor = torch.tensor(kept_indices, device=self.device, dtype=torch.long)
                    current_kept_features = self.original_visual_features[:, kept_indices_tensor, :]
                    current_mean_features = current_kept_features.mean(dim=1)
                    similarity = F.cosine_similarity(self.initial_mean_features, current_mean_features)
                    r_semantic = self.config.GAMMA * similarity.item()
        
        reward = (r_efficiency + r_semantic) / self.current_num_patches
        
        info = {}
        if done:
            final_decisions = self.pruning_mask[:self.current_num_patches]
            kept_indices = torch.tensor(np.where(final_decisions == 1)[0], device=self.device, dtype=torch.long)
            num_kept_final = len(kept_indices)

            if num_kept_final == 0:
                generated_text, is_correct = "", 0.0
            else:
                pruned_visual_features = self.original_visual_features[:, kept_indices, :]
                final_embeddings = torch.cat([self.text_embeds_part1, pruned_visual_features, self.text_embeds_part2], dim=1)
                attention_mask = torch.ones(final_embeddings.shape[:2], dtype=torch.long, device=self.device)
                generated_text = self.mllm.generate_answer(final_embeddings, attention_mask)
                is_correct = 1.0 if self.gt_answer.lower() in generated_text.lower() else 0.0
                # print(f"GT Answer: {self.gt_answer}, Generated: {generated_text}, Correct: {is_correct}")

            r_task = self.config.ALPHA * is_correct
            reward += r_task
            compression_ratio = num_kept_final / self.current_num_patches if self.current_num_patches > 0 else 0.0
            info = {
                "r_task": r_task,
                "num_kept": num_kept_final,
                "answer": generated_text,
                "compression_ratio": compression_ratio,
                "accuracy": is_correct
            }

        return self._get_obs(), reward, done, False, info
