import json

with open("results/benchmarks/batched_1000r_4g/merged_summary.json") as f:
    data = json.load(f)

print("Overall Success Rate:", data.get("success_rate"))
print("Total Successes:", data.get("total_successes"))
print("Verified Successes:", data.get("verified_success"))
print("Extractor Match:", data.get("total_success_extractor"))
print("Ground Truth Match:", data.get("total_success_exact"))
print("Top-1 Success:", data.get("top1_success"))
print("Avg Attempts on Success:", data.get("avg_attempts_on_success"))

print("\nPer Strategy Stats:")
stats = data.get("strategy_stats", {})
for k, v in stats.items():
    print(f"Strategy: {k}")
    print(f"  Attempts: {v.get('attempts')}")
    print(f"  Successes: {v.get('successes')}")
    print(f"  Success Rate: {v.get('success_rate')*100:.2f}%" if v.get('attempts') else "  Success Rate: 0%")

print("\nExtractor Metrics:")
ext = data.get("extractor_metrics", {})
print(json.dumps(ext, indent=2))
