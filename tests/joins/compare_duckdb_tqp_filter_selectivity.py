import json
import os
import matplotlib.pyplot as plt
import numpy as np

result_dir = "./tests/results"
R_values = [0.2, 0.4, 0.6, 0.8, 1.0]
thread_counts = [8, 32]
rows = 10 ** 9

tqp_ours_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"tqp_filter_{rows}rows_{int(R*100)}select.json")
    with open(filename) as f:
        tqp_ours_times.append(json.load(f)["time"])

tqp_ours_no_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"tqp_novortex_filter_{rows}rows_{int(R*100)}select.json")
    with open(filename) as f:
        tqp_ours_no_times.append(json.load(f)["time"])

tqp_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"hand_filter_{rows}rows_{int(R*100)}select.json")
    with open(filename) as f:
        tqp_times.append(json.load(f)["time"])

tqp_cpu_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"hand_filter_cpu_{rows}rows_{int(R*100)}select.json")
    with open(filename) as f:
        tqp_cpu_times.append(json.load(f)["time"])

tqp_kernel_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"hand_filter_gpu_{rows}rows_{int(R*100)}select.json")
    with open(filename) as f:
        tqp_kernel_times.append(json.load(f)["time"])

duckdb_times = {t: [] for t in thread_counts}
for t in thread_counts:
    for R in R_values:
        filename = os.path.join(result_dir, f"duckdb_filter_{t}threads_{int(R*100)}pct_{rows}rows.json")
        with open(filename) as f:
            duckdb_times[t].append(json.load(f)["time"])


plt.figure(figsize=(9, 7))

# max_time = max(max(tqp_times), *(max(v) for v in duckdb_times.values()))
# yticks = np.arange(0, max_time + 0.1, 0.5)  # 每 0.1 秒一个刻度
# plt.yticks(yticks)
plt.xticks(R_values)
# plt.ylim(0, 0.4)

plt.plot(R_values, tqp_ours_times, label="Our TQP (Vortex Enabled)", marker='o', linewidth=2)
plt.plot(R_values, tqp_ours_no_times, label="Our TQP (Vortex Disabled)", marker='o', linewidth=2)
plt.plot(R_values, tqp_times, label="Handwritten PyTorch (CPU storage, GPU compute)", marker='o', linewidth=2)
plt.plot(R_values, tqp_cpu_times, label="Handwritten PyTorch (CPU storage & compute)", marker='o', linewidth=2)
plt.plot(R_values, tqp_kernel_times, label="Handwritten PyTorch (CPU storage, GPU compute), GPU Kernel Time", marker='o', linewidth=2)

colors = ['tab:pink', 'tab:brown']
for idx, t in enumerate(thread_counts):
    # if idx > 0:
    plt.plot(R_values, duckdb_times[t], label=f"DuckDB ({t} threads)", marker='s', linestyle='solid', color=colors[idx])

plt.xlabel("Selectivity")
plt.ylabel("Filter Time (seconds)")
plt.title("Filter Time vs. Selectivity (TQP vs. DuckDB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"filter_rows{rows}.png")
plt.show()
