import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLAMA_PATH = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
device = "cuda:0"

print("Loading...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.float16,
    device_map={"": device},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    ),
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompts = ["Hello, this is a test. " * 100] * 16
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

print(f"Inputs shape: {inputs['input_ids'].shape}")

print("Warming up...")
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=10, do_sample=False)

print("Benchmarking...")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=100, do_sample=False)
torch.cuda.synchronize()
t1 = time.time()
print(f"100 tokens batch=16 took {t1 - t0:.2f}s")
