# GEMINI.md: AutoRed Experiments

## 1. Overview

The `experiment` directory contains a collection of Jupyter notebooks designed to run the AutoRed attack scenarios against various Large Language Models (LLMs) and to analyze the results.

## 2. Directory Structure

```
experiment/
├── gpt_35_turbo.ipynb
├── internlm.ipynb
├── llama_2_7b.ipynb
├── llama_3_8b.ipynb
├── mistral.ipynb
└── result_analysis.ipynb
```

## 3. Running Experiments

Each of the primary notebooks in this directory is set up to run the AutoRed framework against a specific LLM.

### 3.1. Experiment Notebooks

*   **`gpt_35_turbo.ipynb`**: Runs the attack scenario against OpenAI's GPT-3.5-turbo model.
*   **`internlm.ipynb`**: Runs the attack scenario against the InternLM model.
*   **`llama_2_7b.ipynb`**: Runs the attack scenario against the Llama 2 7B model.
*   **`llama_3_8b.ipynb`**: Runs the attack scenario against the Llama 3 8B model.
*   **`mistral.ipynb`**: Runs the attack scenario against the Mistral model.

### 3.2. How to Run an Experiment

To run an experiment, you will need to have a Jupyter environment set up in your virtual environment:

1.  **Install Jupyter:**
    ```bash
    pip install jupyter
    ```

2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

3.  **Open and Run a Notebook:**
    Navigate to the `experiment` directory in the Jupyter interface and open one of the experiment notebooks (e.g., `gpt_35_turbo.ipynb`).

4.  **Configure API Keys and Model Paths:**
    Inside the notebook, you will need to configure the necessary API keys (for models like GPT-3.5-turbo) and/or provide the paths to the locally hosted models (for models like Llama).

5.  **Execute the Cells:**
    Run the cells in the notebook sequentially to execute the attack scenario. The notebook will load the trained AutoRed components (Stop Point Identifier, Malicious Prompt Generator, and Sensitive Information Extractor) and use them to interact with the target LLM.

## 4. Analyzing Results

*   **`result_analysis.ipynb`**: This notebook is used to analyze the results of the experiments. It provides tools to load the logs from the experiment runs, calculate metrics, and visualize the performance of the AutoRed framework against the different LLMs.

After running one or more experiments, you can use this notebook to gain insights into the effectiveness of the attacks and the vulnerabilities of the target models.
