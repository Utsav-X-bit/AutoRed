import json

with open("results/benchmarks/batched_100r_4g/merged_summary.json", "r") as f:
    data = json.load(f)

runs = data.get("runs", [])
total_runs = len(runs)
success_count = sum(1 for r in runs if r.get("verified_success", False))
ground_truth_leaked = sum(1 for r in runs if r.get("ground_truth_leaked", False))
generator_success = sum(1 for r in runs if r.get("generator_success", False))

total_interactions = sum(r.get("interactions_taken", 0) for r in runs)
avg_interactions = total_interactions / total_runs if total_runs > 0 else 0

total_time = sum(r.get("time_taken", 0) for r in runs)
# since it's 4 workers, the actual wall time is max time per worker. Let's just output average time per run.
avg_time_per_run = total_time / total_runs if total_runs > 0 else 0

print(f"Total Scenarios: {total_runs}")
print(f"Verified Successes: {success_count} ({success_count/total_runs*100:.1f}%)")
print(f"Ground Truth Leaked: {ground_truth_leaked} ({ground_truth_leaked/total_runs*100:.1f}%)")
print(f"Generator Success (Extraction): {generator_success} ({generator_success/total_runs*100:.1f}%)")
print(f"Average Interactions: {avg_interactions:.1f}")
print(f"Average Time per Run: {avg_time_per_run:.1f} seconds")

# Compute max wall time
# Group by worker to find max time.
worker_times = {}
for r in runs:
    w = r.get("worker_id", "unknown")
    worker_times[w] = worker_times.get(w, 0) + r.get("time_taken", 0)
    
print("Approximate wall time per worker (sum of runs, real wall time may be lower due to batching overlapping run time tracking):")
for w, t in worker_times.items():
    print(f"  Worker {w}: {t/60:.1f} minutes")

