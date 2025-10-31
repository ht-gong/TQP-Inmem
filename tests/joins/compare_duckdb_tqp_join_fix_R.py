import matplotlib.pyplot as plt
import json
import os
import numpy as np

rows_list = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
thread_counts = [32]
Rs = [1]
results_dir = "./tests/results"

def load_duckdb_results(R):
    duckdb_times = {t: [] for t in thread_counts}
    for t in thread_counts:
        for rows in rows_list:
            filename = f"duckdb_{R}repeats_{t}threads_{rows}rows.json"
            with open(os.path.join(results_dir, filename)) as f:
                data = json.load(f)
                duckdb_times[t].append(data['time'])
    return duckdb_times

def load_tqp_naive_results(R):
    tqp_times = []
    for rows in rows_list:
        filename = f"tqp_newjoin_naive_{R}repeats_{rows}rows.json"
        with open(os.path.join(results_dir, filename)) as f:
            data = json.load(f)
            tqp_times.append(data['time'])
    return tqp_times

def load_tqp_results(R):
    tqp_times = []
    for rows in rows_list:
        filename = f"tqp_newjoin_vortex_{R}repeats_{rows}rows.json"
        with open(os.path.join(results_dir, filename)) as f:
            data = json.load(f)
            tqp_times.append(data['time'])
    return tqp_times



for R in Rs:
    duckdb_times = load_duckdb_results(R)
    tqp_times = load_tqp_results(R)
    tqp_naive_times = load_tqp_naive_results(R)

    plt.figure(figsize=(8, 5))
    plt.plot(rows_list, tqp_times, label='TQP (Vortex Enabled)', marker='o')
    plt.plot(rows_list, tqp_naive_times, label='TQP (Vortex Disabled)', marker='o')
    for t in thread_counts:
        plt.plot(rows_list, duckdb_times[t], label=f'DuckDB {t} threads', marker='s')
    

    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(rows_list, [f"$10^{int(np.log10(x))}$" for x in rows_list])
    plt.xlabel("Input Size (rows)")
    plt.ylabel("Time (s)")
    plt.title(f"Join Performance (R={R})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"join_scale.png")
    plt.show()
