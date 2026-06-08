# GEMINI.md: RL4LMs in AutoRed

## 1. Overview

The `rl4lms` directory contains a forked version of the `RL4LMs` library from the Allen Institute for AI (AI2). This library provides the core components for training and fine-tuning Language Models (LMs) using Reinforcement Learning (RL). In the context of AutoRed, it is used to train the **Malicious Prompt Generator**.

## 2. Directory Structure

The `rl4lms` directory is organized as follows:

```
rl4lms/
├── algorithms/          # RL algorithms (PPO, A2C, NLPO, TRPO)
├── core_components/     # Core components for RL training (sampler, sweep)
├── data_pools/          # Data loading and processing
├── envs/                # Environment setup for text generation tasks
└── __init__.py
```

## 3. Core Components

### 3.1. Algorithms

The `algorithms` directory contains implementations of various RL algorithms. For AutoRed, the key algorithms are:

*   **PPO (Proximal Policy Optimization)**: A policy gradient method that is used for training the Malicious Prompt Generator.
*   **NLPO (Natural Language Policy Optimization)**: Another policy optimization algorithm suitable for language generation tasks.
*   **A2C (Advantage Actor-Critic)**: A synchronous, deterministic actor-critic algorithm.
*   **TRPO (Trust Region Policy Optimization)**: A policy optimization algorithm that enforces a trust region to ensure stable updates.

### 3.2. Environments

The `envs` directory defines the environment for the RL agent to interact with. For AutoRed, the primary environment is `text_generation`, which is tailored for tasks involving text generation and evaluation. It includes components for:

*   **Action and Observation Spaces**: Defining the possible actions the agent can take (generating tokens) and the observations it receives.
*   **Reward Functions**: Calculating rewards based on the generated text. This is a crucial component for guiding the RL training process.
*   **Metrics**: Evaluating the performance of the generated text using metrics like BLEU, ROUGE, and the SPICE metric, which requires the Stanford CoreNLP models (see setup instructions in the root `GEMINI.md`).

### 3.3. Data Pools

The `data_pools` directory is responsible for managing the data used for training. It includes classes for creating, sampling, and processing text-based datasets for use in the RL training loop.

## 4. Role in AutoRed

The `rl4lms` library is fundamental to the training of the **Malicious Prompt Generator**. It provides the necessary tools to:

1.  **Fine-tune LMs**: Start with a pre-trained LM and fine-tune it on a specific task.
2.  **Apply RL**: Use RL algorithms like PPO to optimize the LM for generating malicious prompts that are effective at tricking the target LLM into revealing sensitive information.
3.  **Evaluate Generations**: Use a suite of metrics to evaluate the quality and effectiveness of the generated prompts.

For details on how the models are trained using this library, refer to the documentation in the `scripts` directory.
