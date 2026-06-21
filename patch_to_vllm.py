import sys

with open("experiment/llama_3_8b_verbose.py", "r") as f:
    code = f.read()

# 1. Imports
import_orig = """from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertForSequenceClassification,
)"""
import_new = """from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertForSequenceClassification,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest"""
code = code.replace(import_orig, import_new)

# 2. Victim Model
victim_orig = """    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_PATH,
        local_files_only=True,
        use_fast=False,
    )"""
victim_new = """    llama_model = LLM(
        model=LLAMA_PATH,
        gpu_memory_utilization=0.40,
        tensor_parallel_size=1,
        max_model_len=2048,
        enforce_eager=False,
    )
    llama_tokenizer = llama_model.get_tokenizer()"""
code = code.replace(victim_orig, victim_new)

# 3. chat_with_llama_batch
chat_batch_orig = """    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    original_padding_side = llama_tokenizer.padding_side
    llama_tokenizer.padding_side = "left"

    inputs = llama_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"    [DEBUG] chat_with_llama_batch: generating for {len(attacks)} attacks...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    print(f"    [DEBUG] chat_with_llama_batch: generation complete in {time.time() - t0:.2f}s.", flush=True)

    llama_tokenizer.padding_side = original_padding_side

    responses = []
    for i in range(len(prompts)):
        prompt_len = inputs["input_ids"].shape[1]
        raw_response = llama_tokenizer.decode(
            outputs[i][prompt_len:], skip_special_tokens=True
        )
        responses.append(raw_response.strip())"""
chat_batch_new = """    print(f"    [DEBUG] chat_with_llama_batch: generating for {len(attacks)} attacks...", flush=True)
    t0 = time.time()
    sampling_params = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.9)
    outputs = llama_model.generate(prompts, sampling_params, use_tqdm=False)
    responses = [out.outputs[0].text.strip() for out in outputs]
    print(f"    [DEBUG] chat_with_llama_batch: generation complete in {time.time() - t0:.2f}s.", flush=True)"""
code = code.replace(chat_batch_orig, chat_batch_new)

# 4. load_generator PEFT
peft_orig = """        tokenizer_path = (
            ckpt_path
            if (Path(ckpt_path) / "tokenizer_config.json").exists()
            else base_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, use_fast=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
            attn_implementation="sdpa",
        )
        # Clean max_length from base model BEFORE PeftModel wraps it
        if hasattr(base_model.config, "max_length"):
            delattr(base_model.config, "max_length")
        if hasattr(base_model, "generation_config") and hasattr(base_model.generation_config, "max_length"):
            base_model.generation_config.max_length = None
        model = PeftModel.from_pretrained(base_model, ckpt_path)"""
peft_new = """        global gen_lora_request
        gen_lora_request = LoRARequest("generator_adapter", 1, ckpt_path)
        model = LLM(
            model=base_model_path,
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=0.40,
            tensor_parallel_size=1,
            max_model_len=2048,
        )
        tokenizer = model.get_tokenizer()"""
code = code.replace(peft_orig, peft_new)

# 5. load_generator Base
gen_base_orig = """        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, local_files_only=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
            attn_implementation="sdpa",
        )"""
gen_base_new = """        global gen_lora_request
        gen_lora_request = None
        model = LLM(
            model=ckpt_path,
            gpu_memory_utilization=0.40,
            tensor_parallel_size=1,
            max_model_len=2048,
        )
        tokenizer = model.get_tokenizer()"""
code = code.replace(gen_base_orig, gen_base_new)

# 6. inference_gen_model_verbose_batch
inf_batch_orig = """    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    original_padding_side = gen_tokenizer.padding_side
    gen_tokenizer.padding_side = "left"

    inputs = gen_tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"    [DEBUG] inference_gen_model_verbose_batch: generating for {len(prompt_texts)} prompts...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    print(f"    [DEBUG] inference_gen_model_verbose_batch: generation complete in {time.time() - t0:.2f}s.", flush=True)

    gen_tokenizer.padding_side = original_padding_side

    responses = []
    for i in range(len(prompt_texts)):
        prompt_len = inputs["input_ids"].shape[1]
        raw_response = gen_tokenizer.decode(
            outputs[i][prompt_len:], skip_special_tokens=True
        )
        responses.append(raw_response.strip())"""
inf_batch_new = """    global gen_lora_request
    print(f"    [DEBUG] inference_gen_model_verbose_batch: generating for {len(prompt_texts)} prompts...", flush=True)
    t0 = time.time()
    sampling_params = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.9)
    outputs = gen_model.generate(prompt_texts, sampling_params, use_tqdm=False, lora_request=gen_lora_request)
    responses = [out.outputs[0].text.strip() for out in outputs]
    print(f"    [DEBUG] inference_gen_model_verbose_batch: generation complete in {time.time() - t0:.2f}s.", flush=True)"""
code = code.replace(inf_batch_orig, inf_batch_new)

# 7. extract_batch
extract_orig = """        if tkr.pad_token is None:
            tkr.pad_token = tkr.eos_token
        orig_padding = tkr.padding_side
        tkr.padding_side = "left"

        inputs = tkr(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"    [DEBUG] extract_batch: verifying {len(prompts)} candidates...", flush=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_new_tokens=180, do_sample=False)
        print(f"    [DEBUG] extract_batch: verification complete in {time.time() - t0:.2f}s.", flush=True)

        tkr.padding_side = orig_padding

        llm_raw_outputs = []
        for i in range(len(prompts)):
            prompt_len = inputs["input_ids"].shape[1]
            raw = tkr.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
            llm_raw_outputs.append(raw)"""
extract_new = """        print(f"    [DEBUG] extract_batch: verifying {len(prompts)} candidates...", flush=True)
        t0 = time.time()
        sampling_params = SamplingParams(max_tokens=180, temperature=0.0)
        outputs = mdl.generate(prompts, sampling_params, use_tqdm=False)
        print(f"    [DEBUG] extract_batch: verification complete in {time.time() - t0:.2f}s.", flush=True)

        llm_raw_outputs = [out.outputs[0].text.strip() for out in outputs]"""
code = code.replace(extract_orig, extract_new)

# Increase batch size for benchmark
code = code.replace("BATCH_SIZE = 10", "BATCH_SIZE = 50")

with open("experiment/llama_3_8b_vllm.py", "w") as f:
    f.write(code)
print("vLLM Patch Complete!")
