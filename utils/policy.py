import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy
from tianshou.data import Batch

class CompositeDistribution:
    """
    A custom distribution class for our composite action space.
    Implements the full interface expected by Tianshou's framework.
    """
    ndim = 1

    def __init__(self, dist_select, dist_keep):
        self.dist_select: torch.distributions.Categorical = dist_select
        self.dist_keep: torch.distributions.Bernoulli = dist_keep
        self._batch_size = len(self.dist_select.logits)
        self._num_patches = self.dist_select.logits.shape[1]
        self._possible_indices = torch.arange(
            self._num_patches, device=self.dist_select.logits.device
        ).float()

    def __len__(self):
        return self._batch_size

    def sample(self, sample_shape=torch.Size()):
        if len(sample_shape) > 0:
            raise NotImplementedError("Sample shape for composite distribution not implemented")
        index = self.dist_select.sample()
        all_decisions = self.dist_keep.sample()
        batch_indices = torch.arange(self._batch_size, device=index.device)
        specific_decision = all_decisions[batch_indices, index]
        return torch.stack([index.float(), specific_decision], dim=1)

    def log_prob(self, act):
        index = act[:, 0].long()
        decision = act[:, 1]
        batch_indices = torch.arange(self._batch_size, device=index.device)
        log_prob_select = self.dist_select.log_prob(index)
        log_prob_keep_full = self.dist_keep.log_prob(
            decision.unsqueeze(1).expand_as(self.dist_keep.logits)
        )
        log_prob_keep = log_prob_keep_full[batch_indices, index]
        return log_prob_select + log_prob_keep

    def entropy(self):
        entropy_select = self.dist_select.entropy()
        entropy_keep_per_token = self.dist_keep.entropy()
        probs_select = self.dist_select.probs
        entropy_conditional_keep = (probs_select * entropy_keep_per_token).sum(dim=-1)
        return entropy_select + entropy_conditional_keep

    @property
    def mean(self) -> torch.Tensor:
        probs_select = self.dist_select.probs
        mean_select = (probs_select * self._possible_indices).sum(dim=-1)
        probs_keep = self.dist_keep.probs
        mean_keep = (probs_select * probs_keep).sum(dim=-1)
        return torch.stack([mean_select, mean_keep], dim=-1)

    @property
    def stddev(self) -> torch.Tensor:
        probs_select = self.dist_select.probs
        mean_select = (probs_select * self._possible_indices).sum(dim=-1, keepdim=True)
        var_select = (probs_select * (self._possible_indices - mean_select) ** 2).sum(dim=-1)
        std_select = torch.sqrt(var_select)
        probs_keep = self.dist_keep.probs
        std_keep_per_token = torch.sqrt(probs_keep * (1.0 - probs_keep))
        expected_std_keep = (probs_select * std_keep_per_token).sum(dim=-1)
        return torch.stack([std_select, expected_std_keep], dim=-1)


class PolicyValueNet(nn.Module):
    """
    The policy and value network for the PPO agent.
    """
    def __init__(self, vision_dim, text_dim, hidden_dim, max_num_patches):
        super().__init__()
        self.max_num_patches = max_num_patches
        self.fusion_ffn = nn.Sequential(
            nn.Linear(vision_dim + text_dim + vision_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.select_w_h = nn.Linear(hidden_dim, hidden_dim)
        self.select_w_m = nn.Linear(self.max_num_patches, hidden_dim)
        self.select_w = nn.Linear(hidden_dim, 1)
        self.keep_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, state=None, info={}):
        is_batch_object = not isinstance(obs, dict)
        device = next(self.parameters()).device
        
        visual_features = torch.as_tensor(obs.visual_features if is_batch_object else obs["visual_features"], dtype=torch.float32, device=device)
        query_embeddings = torch.as_tensor(obs.query_embeddings if is_batch_object else obs["query_embeddings"], dtype=torch.float32, device=device)
        pruning_mask = torch.as_tensor(obs.pruning_mask if is_batch_object else obs["pruning_mask"], dtype=torch.float32, device=device)
        
        batch_size, num_patches, _ = visual_features.shape
        q_expanded = query_embeddings.squeeze(1).expand(-1, num_patches, -1)
        
        h_odot_q = visual_features * q_expanded
        #fusion_input = torch.cat([visual_features, q_expanded, h_odot_q], dim=-1)
        fusion_input = torch.cat([visual_features, q_expanded], dim=-1)
        hfuse = self.fusion_ffn(fusion_input)
        
        mask_context = self.select_w_m(pruning_mask).unsqueeze(1)
        e_hidden = torch.tanh(self.select_w_h(hfuse) + mask_context)
        
        select_logits = self.select_w(e_hidden).squeeze(-1)
        keep_logits = self.keep_head(hfuse).squeeze(-1)
        
        non_pruned_mask = (pruning_mask != 0).float().unsqueeze(-1)
        num_non_pruned = torch.clamp(non_pruned_mask.sum(dim=1), min=1)
        sum_features = (hfuse * non_pruned_mask).sum(dim=1)
        mean_features = sum_features / num_non_pruned
        value = self.critic_head(mean_features)
        
        return (select_logits, keep_logits), value


class CompositeActionPPO(PPOPolicy):
    """
    A PPO Policy modified to handle the composite action space.
    """
    def forward(self, batch, state=None, **kwargs):
        (select_logits, keep_logits), value = self.actor(batch.obs)
        
        if isinstance(batch.obs, Batch):
            pruning_mask = batch.obs.pruning_mask
            valid_token_mask = batch.obs.valid_token_mask
        else:
            pruning_mask = torch.tensor(batch.obs["pruning_mask"], device=select_logits.device)
            valid_token_mask = torch.tensor(batch.obs["valid_token_mask"], device=select_logits.device)

        undecided_mask = (pruning_mask == -1) & (valid_token_mask == 1)
        select_logits[~undecided_mask] = -float('inf')
        
        dist_select = torch.distributions.Categorical(logits=select_logits)
        dist_keep = torch.distributions.Bernoulli(logits=keep_logits)
        
        dist = CompositeDistribution(dist_select, dist_keep)
        act = dist.sample()
        
        return Batch(act=act, state=state, dist=dist, value=value.flatten())


class CriticWrapper(nn.Module):
    """
    A wrapper for the critic network, ensuring it only returns the value.
    """
    def __init__(self, actor_critic_net):
        super().__init__()
        self.net = actor_critic_net

    def forward(self, obs, *args, **kwargs):
        _, value = self.net(obs, *args, **kwargs)
        return value
