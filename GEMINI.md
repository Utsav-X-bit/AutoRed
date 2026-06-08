# GEMINI.md: AutoRed

## 1. Overview

AutoRed is an automated framework for red teaming Large Language Models (LLMs). It's designed to generate malicious attack scenarios to test and evaluate the security of LLMs, specifically their susceptibility to sensitive information leakage. The framework is built upon the `RL4LMs` library from the Allen Institute for AI (AI2).

The primary goal of AutoRed is to automate the costly and time-consuming process of manual red teaming by using a combination of machine learning models to simulate attacks.

## 2. Architecture

The AutoRed framework consists of three core components that work in concert to generate and execute attack scenarios:

*   **Stop Point Identifier**: A high-level decision-making model. This binary classifier determines the current phase of the attack, deciding whether to generate a malicious prompt or to attempt to extract sensitive information.

*   **Malicious Prompt Generator**: A low-level model responsible for creating diverse and effective prompt injection attacks. This component is trained using a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) techniques.

*   **Sensitive Information Extractor**: Another low-level model, this component is a few-shot engineered GPT-3.5-turbo model specifically designed to extract sensitive data from the target LLM once a vulnerability has been exposed by the Malicious Prompt Generator.

The overall architecture is depicted in the following diagram from the research paper:

![AutoRed Model](assets/autored-model.png)

## 3. Codebase Structure

The AutoRed repository is organized as follows:

```
/
├── assets/                  # Logos and architectural diagrams
├── experiment/              # Jupyter notebooks for running experiments
├── rl4lms/                  # Core library for RL for LMs, forked from AllenAI
│   ├── algorithms/          # RL algorithms (PPO, A2C, etc.)
│   ├── core_components/     # Core components for RL training
│   ├── data_pools/          # Data loading and processing
│   └── envs/                  # Environment setup for text generation tasks
├── scripts/                 # Training and data generation scripts
│   ├── pi/                  # Scripts for the Prompt Injection tasks
│   └── training/            # Scripts for training the models
├── LICENSE                  # License file
├── README.md                # Project README
├── requirements.txt         # Python dependencies
└── setup.py                 # Project setup script
```

## 4. Setup and Installation

### 4.1. Prerequisites

*   Python 3.7+
*   NVIDIA GPU with CUDA support (recommended for training)

### 4.2. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yoyostudy/AutoRed.git
    cd AutoRed
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install PyTorch:**
    The `requirements.txt` file specifies `torch==1.12.0` and `torchvision==0.13.0`. For systems with CUDA, it is recommended to install the appropriate version from the PyTorch website. For example, for CUDA 11.3:
    ```bash
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
    For other CUDA versions or for CPU-only installation, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

4.  **Install the remaining dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install the `autored` package:**
    ```bash
    pip install -e .
    ```
    The `-e` flag installs the package in "editable" mode, which is useful for development.

### 4.3. Stanford CoreNLP Models

The SPICE metric used in the `rl4lms` library requires the Stanford CoreNLP models. To download them, run the following script:

```bash
cd rl4lms/envs/text_generation/caption_metrics/spice
./get_stanford_models.sh
cd ../../../../../../
```

## 5. Running Experiments and Training

The `experiment` directory contains Jupyter notebooks for running experiments with different models. The `scripts` directory contains the necessary scripts for data creation and model training.

For detailed instructions on running the experiments and training the models, please refer to the `GEMINI.md` files in the respective directories:

*   [`scripts/GEMINI.md`](scripts/GEMINI.md): For training and data generation.
*   [`experiment/GEMINI.md`](experiment/GEMINI.md): For running the experiments.
*   [`rl4lms/GEMINI.md`](rl4lms/GEMINI.md): For an overview of the RL4LMs library.
