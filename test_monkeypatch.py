from transformers import AutoTokenizer, PreTrainedTokenizerFast

if not hasattr(PreTrainedTokenizerFast, "all_special_tokens_extended"):
    print("Patching...")
    PreTrainedTokenizerFast.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print(tokenizer.all_special_tokens_extended)
