#!/usr/bin/env python3
import os

FILE_PATH = "experiment/llama_3_8b_vllm.py"

with open(FILE_PATH, "r") as f:
    content = f.read()

# Replacement 1
old1 = """    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)"""
new1 = """    # Save to results directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path("results") / date_str
    results_dir.mkdir(parents=True, exist_ok=True)"""
content = content.replace(old1, new1)

# Replacement 2
old2 = """    # JSON emission: save per-round run JSONs
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)"""
new2 = """    # JSON emission: save per-round run JSONs
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path("results") / date_str
    results_dir.mkdir(parents=True, exist_ok=True)"""
content = content.replace(old2, new2)

# Replacement 3
old3 = """    # Save extractor benchmarking metrics
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)"""
new3 = """    # Save extractor benchmarking metrics
    date_str = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path("results") / date_str
    results_dir.mkdir(parents=True, exist_ok=True)"""
content = content.replace(old3, new3)

with open(FILE_PATH, "w") as f:
    f.write(content)

print("Patch applied successfully.")
