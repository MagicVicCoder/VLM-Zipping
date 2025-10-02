import torch
import os
from datetime import datetime

# --- Global Settings ---
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- MLLM and Dataset Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "lmms-lab/MME"
DATASET_SPLIT = "test[:1000]"  # 增加样本量以获得更可靠的结果
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Environment Hyperparameters ---
# Reward coefficients
ALPHA = 1.0  # Task reward weight
BETA = 0.5   # Efficiency reward weight
GAMMA = 0.1  # Semantic similarity reward weight (稍微降低，更关注任务和效率)

# --- PPO Policy & Trainer Hyperparameters ---
MAX_PATCHES = 2000  # 根据Qwen VL的典型patch数量调整
HIDDEN_DIM = 512
LR = 1e-5  # 稍微降低学习率以增加稳定性
GAMMA_PPO = 0.99  # Discount factor for PPO
EPS_CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
GAE_LAMBDA = 0.95
REWARD_NORMALIZATION = True

# --- Training Loop Settings ---
EPOCHS = 5  # 增加训练轮次以保证收敛
BATCH_SIZE = 16
BUFFER_SIZE = 10000
STEP_PER_COLLECT = 2000  # 增加每次收集的步数
STEP_PER_EPOCH = 8000    # 增加每轮的步数
REPEAT_PER_COLLECT = 2
NUM_TRAIN_ENVS = 10  # 使用多个环境并行采样，加快训练速度
NUM_TEST_ENVS = 1
EPISODE_PER_TEST = 10 # 增加测试时的episode数量

# --- Inference Settings ---
DETERMINISTIC_INFERENCE = True

# --- Evaluation Settings ---
EVAL_MODE = "full"  # 可选值："full", "budget", "none"
EVAL_BUDGET_RATIO = 0.5  # 仅在 EVAL_MODE=="budget" 时生效，表示预算比例（0~1）