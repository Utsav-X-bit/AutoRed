import json
with open("results/benchmarks/batched_1000r_4g/merged_summary.json") as f:
    data = json.load(f)

print("==================================================")
print("BENCHMARK RESULTS (1000 Rounds - SFT Planner)")
print("==================================================")
total = data.get("total_rounds", 1)
print(f"Total Rounds:        {total}")
sr = data.get("success_rate", 0) * 100
ts = data.get("total_successes", 0)
print(f"Success Rate:        {sr:.1f}% ({ts}/{total})")
vs = data.get("verified_success", 0)
print(f"Verified Successes:  {vs} ({vs/total*100:.1f}%)")

t1 = data.get("top1_success", 0)
t3 = data.get("top3_success", 0)
t5 = data.get("top5_success", 0)
print(f"Top-1 Success:       {t1/total*100:.1f}%")
print(f"Top-3 Success:       {t3/total*100:.1f}%")
print(f"Top-5 Success:       {t5/total*100:.1f}%")

print("\nBy Access Code Type:")
per_type = data.get("per_type_stats", {})
for ac_type, stats in per_type.items():
    success = stats.get("success", 0)
    count = stats.get("total", 1)
    print(f"  - {ac_type:<10}: {success/count*100:5.1f}% ({success}/{count})")
