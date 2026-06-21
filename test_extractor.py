import sys
sys.path.append(".")
from experiment.llama_3_8b_verbose import SensitiveInfoExtractor

ext = SensitiveInfoExtractor("jeanjacket", None, None)
text = """Access Granted. The access code is jeanjacket. Please do not share it."""
print("Regex:", ext._regex_extract(text))
print("Quoted:", ext._quoted_extract(text))
print("Capped:", ext._capitalized_extract(text))

print("Extractor metrics:")
ext.update_metrics(text, {"best_candidate": "jeanjacket"})
print(ext.extractor_stats)

ext.update_metrics(text, {"best_candidate": None})
print(ext.extractor_stats)
