import json
import os
import matplotlib.pyplot as plt
import numpy as np

result_dir = "./tests/results"
R_values = [1, 5, 10, 20]  
thread_counts = [1, 8, 32]
rows = 10 ** 8

tqp_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"tqp_{R}repeats_{rows}rows.json")
    with open(filename) as f:
        tqp_times.append(json.load(f)["time"])

tqp_vortex_times = []
for R in R_values:
    filename = os.path.join(result_dir, f"tqp_vortex_{R}repeats_{rows}rows.json")
    with open(filename) as f:
        tqp_vortex_times.append(json.load(f)["time"])

duckdb_times = {t: [] for t in thread_counts}
for t in thread_counts:
    for R in R_values:
        filename = os.path.join(result_dir, f"duckdb_{R}repeats_{t}threads_{rows}rows.json")
        with open(filename) as f:
            duckdb_times[t].append(json.load(f)["time"])


plt.figure(figsize=(9, 5))

# max_time = max(max(tqp_times), *(max(v) for v in duckdb_times.values()))
# yticks = np.arange(0, max_time + 0.1, 0.5)  # 每 0.1 秒一个刻度
# plt.yticks(yticks)
plt.xticks(R_values)

plt.plot(R_values, tqp_times, label="TQP (Vortex Disabled)", marker='o', linewidth=2)
plt.plot(R_values, tqp_vortex_times, label="TQP (Vortex Enabled)", marker='o', linewidth=2)

colors = ['tab:red', 'tab:green', 'tab:pink']
for idx, t in enumerate(thread_counts):
    plt.plot(R_values, duckdb_times[t], label=f"DuckDB ({t} threads)", marker='s', linestyle='--', color=colors[idx])

plt.xlabel("Repetition Factor R")
plt.ylabel("Join Time (seconds)")
plt.title("Join Time vs. R (TQP vs. DuckDB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"join_rows{rows}.png")
plt.show()
