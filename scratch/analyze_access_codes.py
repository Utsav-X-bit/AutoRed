import json
import re

with open("results/benchmarks/batched_1000r_4g/merged_summary.json") as f:
    data = json.load(f)

results = data.get("results", [])

def classify_code(code: str) -> str:
    if not code: return "Empty"
    code = code.strip()
    if re.search(r'^[a-zA-Z0-9]{16,}$', code): return "Long Alphanumeric"
    if re.match(r'^[a-zA-Z0-9]+$', code): return "Alphanumeric Word"
    if " " in code and len(code.split()) > 3: return "Phrase/Sentence"
    if re.search(r'[!@#$%^&*()_+={}\[\]|\\:;"\'<>,.?/]', code): return "Contains Special Chars"
    return "Other"

stats_all = {}
stats_success = {}

for r in results:
    code = r.get("access_code", "")
    success = r.get("success", False)
    
    ctype = classify_code(code)
    stats_all[ctype] = stats_all.get(ctype, 0) + 1
    if success:
        stats_success[ctype] = stats_success.get(ctype, 0) + 1

print("--- Success Rate by Access Code Pattern ---")
for ctype, count in stats_all.items():
    suc = stats_success.get(ctype, 0)
    print(f"{ctype:25}: {suc:3}/{count:3} ({(suc/count)*100:.1f}%)")

print("\n--- Why is the Generator producing fewer results? ---")
print("Total Successes: 380 (38.0%)")
print("Ground Truth Matches (Extracted exactly): 377")
print("Verified Successes: 209 (20.9%)")
print("\nExtractor Metrics:")
print(json.dumps(data.get("extractor_metrics", {}), indent=2))
