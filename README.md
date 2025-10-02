# 代码结构

```
.
├── config.py               # 配置文件，包含所有超参数和设置
├── main.py                 # 主脚本，用于运行训练和评估流程
├── data/
│   ├── base_loader.py      # 数据加载器的基类
│   └── vqa_loader.py       # 测试用MME数据加载器
├── models/
│   ├── base_mllm.py        # MLLM 的基类
│   └── qwen_mllm.py        # Qwen MLLM 的实现
├── trainer/
│   └── trainer.py          # 包含 RL 训练逻辑 (PPO)
├── evaluator/
│   └── evaluator.py        # 包含对已训练策略的评估逻辑
├── utils/
│   ├── env.py              # RL 环境定义
│   └── policy.py           # PPO 策略网络定义
├── logs/                   # 用于存放日志文件的目录
└── README.md               # 本文件
```

# 如何适配新数据集

要将此框架适配到新的视觉问答（VQA）数据集，请按照以下步骤操作：

1.  **创建新的数据加载器**:
    - 在 `data/` 创建一个新的文件，创建新类，继承自 `data.base_loader.BaseLoader` 。
    - 实现 `_load_and_split_data` 方法，该方法应加载你的数据集并把训练和测试样本保存为类的属性。样本保存为字典，包含 `image`（PIL.Image 对象）、`question`（字符串）和 `answer`（字符串）。

2.  **集成新的加载器**:
    - 打开 `data/base_loader.py` 文件。
    - 在 `get_data_loader` 函数中，添加一个条件分支，当 `config.DATASET_NAME` 匹配你的数据集名称时，实例化你新创建的数据加载器。

3.  **更新配置**:
    - 在 `config.py` 中，将 `DATASET_NAME` 的值更改为你的数据集名称。
    - 根据需要设置参数。

# 如何适配新模型

要集成一个新的多模态大语言模型（MLLM），请执行以下步骤：

1.  **创建新的模型封装类**:
    - 在 `models/` 目录下创建一个新的 Python 文件。
    - 在该文件中，创建一个继承自 `models.base_mllm.BaseMLLM` 的新类。
    - 实现以下关键方法：
        - `__init__(self, config)`: 加载指定 `MODEL_ID` 的模型和分词器。
        - `get_components_for_env(self, image, question)`: 处理输入的图像和问题，并返回一个字典，其中包含 RL 环境所需的所有组件。包括：
            - `original_visual_features`: 原始的视觉特征。
            - `query_embeddings`: 问题相关的文本嵌入。
            - `text_embeds_part1` / `text_embeds_part2`: 图像嵌入前后，用于在剪枝后重建最终输入序列的文本嵌入部分。
            - `current_num_patches`: 图像中的 patch 数量。
        - `generate_answer(self, final_embeddings, attention_mask)`: 根据经过剪枝后重组的最终嵌入序列生成答案。

2.  **集成新的模型**:
    - 打开 `models/base_mllm.py` 文件。
    - 在 `get_mllm` 函数中，添加一个条件分支，当 `config.MODEL_ID` 匹配你的模型 ID 时，实例化你新创建的模型封装类。

3.  **更新配置**:
    - 在 `config.py` 中，将 `MODEL_ID` 的值更改为你的模型在 Hugging Face Hub 上的标识符。

# 超参数说明

所有重要的超参数都在 `config.py` 文件中定义：

## 通用参数
-   `MODEL_ID`: 使用的 MLLM 的 Hugging Face 标识符。
-   `DATASET_NAME`: 使用的数据集名称。
-   `MAX_PATCHES`: 模型可以处理的最大 patch 数量。超过此数量的 patch 将被截断。

## RL 环境参数
-   `ALPHA`, `BETA`, `GAMMA`: RL 环境中奖励函数的权重系数。
    - `ALPHA`: 任务奖励（例如，VQA 准确率）的权重。
    - `BETA`: 效率奖励（与剪枝率相关）的权重。
    - `GAMMA`: 语义相似度奖励的权重（可选）。

## PPO 训练参数
-   `HIDDEN_DIM`: PPO 策略网络中隐藏层的维度。
-   `LR`: PPO 优化器的学习率。
-   `GAMMA_PPO`: PPO 的折扣因子。
-   `EPS_CLIP`: PPO 中的裁剪范围。
-   `VF_COEF`: 值函数损失的系数。
-   `ENT_COEF`: 熵奖励的系数。
-   `GAE_LAMBDA`: GAE（广义优势估计）中的 Lambda 参数。
-   `REWARD_NORMALIZATION`: 是否对奖励进行归一化。
-   `EPOCHS`: 训练的总轮次。
-   `BATCH_SIZE`: 训练时的批次大小。
-   `BUFFER_SIZE`: 经验回放缓冲区的大小。
-   `STEP_PER_COLLECT`: 每次数据收集的步数。
-   `STEP_PER_EPOCH`: 每个训练轮次的总步数。
-   `REPEAT_PER_COLLECT`: 每次收集后，策略更新的次数。
-   `NUM_TRAIN_ENVS`: 用于并行采样的训练环境数量。

### 评估参数
-   `EVAL_MODE`: 评估模式，可选值为：
    - `"full"`: 对所有有效的 token 根据阈值进行决策。
    - `"budget"`: 仅对概率最高的 `EVAL_BUDGET_RATIO` 比例的 token 进行决策。
    - `"none"`: 不进行任何剪枝，作为性能基线。
-   `EVAL_BUDGET_RATIO`: 在 `"budget"` 模式下使用的预算比例。
-   `THRESHOLD`: 用于决定是否保留一个 token 的概率阈值。
