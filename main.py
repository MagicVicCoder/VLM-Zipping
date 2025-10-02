import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import logging

# @param {type:"string"}
import config
from data.base_loader import get_data_loader
from models.base_mllm import get_mllm
from trainer.trainer import setup_environments, setup_policy, train_agent
from evaluator.evaluator import evaluate_performance

def setup_logger():
    """Sets up the logger to write to a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(config.LOG_FILE)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main():
    """
    Main function to run the complete RL-based token pruning pipeline.
    """
    # 0. Setup Logger
    logger = setup_logger()
    logger.info("--- 0. Logger Initialized ---")

    # 1. Load Data
    logger.info("--- 1. Initializing Data Loader ---")
    data_loader = get_data_loader(config)
    logger.info(f"Data loader for '{config.DATASET_NAME}' initialized.")

    # 2. Load MLLM
    logger.info("--- 2. Initializing MLLM ---")
    mllm = get_mllm(config)
    logger.info(f"MLLM '{config.MODEL_ID}' initialized.")

    # 3. Setup RL Environment and Policy
    logger.info("--- 3. Setting up RL Environment and Policy ---")
    train_envs, test_envs = setup_environments(config, mllm, data_loader)
    policy = setup_policy(config, mllm, train_envs)
    logger.info("Environments and PPO policy are ready.")

    # 4. Train the Agent
    logger.info("--- 4. Starting Agent Training ---")
    trained_policy = train_agent(config, policy, train_envs, test_envs)
    logger.info("Agent training finished.")
    
    # 5. Evaluate the Agent
    logger.info("--- 5. Starting Agent Evaluation ---")
    config.EVAL_MODE = "none"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)
    config.EVAL_MODE = "full"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)
    config.EVAL_MODE = "budget"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)
    logger.info("Agent evaluation finished.")

if __name__ == "__main__":
    main()