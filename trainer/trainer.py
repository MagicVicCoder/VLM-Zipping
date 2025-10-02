import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer as onpolicy_trainer
from utils.env import MLLMTokenPruningEnv
from utils.policy import PolicyValueNet, CompositeActionPPO, CriticWrapper

def setup_environments(config, mllm, data_loader):
    """
    Create training and testing environments.
    """
    train_samples = data_loader.get_train_samples()
    test_samples = data_loader.get_test_samples()

    def make_train_env():
        return MLLMTokenPruningEnv(mllm, train_samples, config, np.random.randint(0, 1e6))

    def make_test_env():
        return MLLMTokenPruningEnv(mllm, test_samples, config, np.random.randint(0, 1e6))

    train_envs = DummyVectorEnv([make_train_env for _ in range(config.NUM_TRAIN_ENVS)])
    test_envs = DummyVectorEnv([make_test_env for _ in range(config.NUM_TEST_ENVS)])
    
    return train_envs, test_envs

def setup_policy(config, mllm, train_envs):
    """
    Initialize the PPO policy and optimizer.
    """
    net = PolicyValueNet(
        vision_dim=mllm.feature_dim,
        text_dim=mllm.feature_dim,
        hidden_dim=config.HIDDEN_DIM,
        max_num_patches=config.MAX_PATCHES
    ).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    critic_net = CriticWrapper(net)
    
    policy = CompositeActionPPO(
        actor=net,
        critic=critic_net,
        optim=optimizer,
        dist_fn=lambda *args, **kwargs: torch.distributions.Distribution(),
        action_space=train_envs.action_space[0] if isinstance(train_envs.action_space, list) else train_envs.action_space,
        eps_clip=config.EPS_CLIP,
        discount_factor=config.GAMMA_PPO,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        gae_lambda=config.GAE_LAMBDA,
        reward_normalization=config.REWARD_NORMALIZATION,
        action_scaling=False
    ).to(config.DEVICE)
    policy.device = config.DEVICE
    
    return policy

def train_agent(config, policy, train_envs, test_envs):
    """
    Run the Tianshou on-policy trainer.
    """
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(config.BUFFER_SIZE, len(train_envs)))
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    print("\nStarting Tianshou On-policy Trainer...")
    
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config.EPOCHS,
        step_per_epoch=config.STEP_PER_EPOCH,
        repeat_per_collect=config.REPEAT_PER_COLLECT,
        episode_per_test=config.EPISODE_PER_TEST,
        batch_size=config.BATCH_SIZE,
        step_per_collect=config.STEP_PER_COLLECT,
        verbose=True,
    ).run()
    
    print(f"\nFinished training! Result:\n{result}")
    return policy
