import argparse
import json
import os
import re
import sys
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PCIE_SPEED_GBS = 25 * 1e9
GPU_MEM_SPEED_GBS = 1500 * 1e9
ZERO_CPY_SPEED_GBS = PCIE_SPEED_GBS * 4

def load_json_total_breakdown(prefix, SF, qid, modes):
    """
    Parse result JSONs, augment them with theoretical I/O times,
    and write those times to a CSV.

    CSV schema:
        query,runtime_s
        1,0.0234
        2,0.0198
        ...
    """
    results = {}
    csv_rows = []                     # <- collect rows here

    for mode in modes:
        cpu_in_tots, cpu_out_tots, gpu_tots, zero_cpy_tots = ([] for _ in range(4))

        for Q in qid:
            path = f"experiments/results/{prefix}/{mode}/{SF}_Q{Q}.json"
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}", file=sys.stderr)
                zero_cpy_tots.append(cpu_in_tots.append(cpu_out_tots.append(gpu_tots.append(0) or 0) or 0) or 0)
                continue

            cpu_in = cpu_out = gpu = zero_cpy =  0
            for inner in data.values():
                if not isinstance(inner, dict):
                    continue
                for metric, val in inner.items():
                    if re.search(r"Naive CPU In", metric):
                        cpu_in += val
                    elif re.search(r".* CPU Out", metric):
                        cpu_out += val
                    elif re.search(r".* GPU .*", metric):
                        gpu += val
                    elif re.search(r".*Zero Copy .*", metric):
                        zero_cpy += val

            # aggregate per‑query totals
            cpu_in_tots.append(cpu_in)
            cpu_out_tots.append(cpu_out)
            gpu_tots.append(gpu)
            zero_cpy_tots.append(zero_cpy)

            no_overlap_theoretical = (cpu_in + cpu_out) / PCIE_SPEED_GBS + gpu / GPU_MEM_SPEED_GBS + zero_cpy / ZERO_CPY_SPEED_GBS
            half_overlap_theoretical = (
                (cpu_in * 0.5 + max(cpu_in * 0.5, cpu_out * 0.5) + cpu_out * 0.5) / PCIE_SPEED_GBS
                + gpu / GPU_MEM_SPEED_GBS + zero_cpy / ZERO_CPY_SPEED_GBS
            )

            # add to JSON
            data["no_overlap_theoretical_IO_time_s"] = no_overlap_theoretical
            data["half_overlap_theoretical_IO_time_s"] = half_overlap_theoretical
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            # collect for CSV
            csv_rows.append({"query": Q, "runtime_s": no_overlap_theoretical})

        results[mode] = {
            "CPU in": np.array(cpu_in_tots),
            "GPU": np.array(gpu_tots),
            "CPU out": np.array(cpu_out_tots),
            "Zero Copy in": np.array(zero_cpy_tots)
        }
    # ---------- write CSV ----------
    csv_path = Path(f"experiments/results/{prefix}/{SF}_no_overlap_io_times.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "runtime_s"])
        writer.writeheader()
        writer.writerows(csv_rows)

    return results

def plot_stacked_bar(queries, results, config, title, outfile, modes, legend_loc='upper right'):
    all_modes = modes
    num_bars  = len(all_modes)
    bar_w     = 0.8 / num_bars
    offsets   = {m: bar_w * (i - (num_bars - 1) / 2)
                 for i, m in enumerate(all_modes)}
    x = np.arange(len(queries))

    plt.figure(figsize=(14, 6))

    for mode in modes:
        bottom = np.zeros(len(queries), dtype=float)
        for lbl, key, color in config:
            vals = results[mode][key] / 1e9
            bars = plt.bar(
                x + offsets[mode], vals, bar_w, bottom=bottom,
                label=lbl if mode == modes[0] else None,
                color=color, edgecolor='black', linewidth=0.2
            )

            # --- annotate every bar segment -------------------------
            for rect, val in zip(bars, vals):
                if val == 0:         # skip empty segments
                    continue
                y = rect.get_y() + rect.get_height() / 2
                plt.text(rect.get_x() + rect.get_width() / 2,
                         y, f'{val :.2f}',
                         ha='center', va='center', fontsize=8)

            bottom += vals

    plt.xticks(x, queries, rotation=45)
    plt.ylabel("Total data movement (GB)")
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()

def main():
    # qid   = list(range(1, 23))
    qid = [1, 3,4,5,6,8, 10, 12, 13, 14, 15, 17, 19, 20, 22]
    prefix   = "2025-09-16"
    SF    = "SF300"
    modes = ['datasize']
    times = {mode: {} for mode in modes}
    queries = [f"Q{q}" for q in qid]

    breakdowns = [
        (load_json_total_breakdown,  [
            ('CPU -> GPU', 'CPU in', "#F5DA0D"),
            ('GPU -> CPU', 'CPU out', "#EEB20B"),
            ('GPU -> GPU', 'GPU', "#F56A0D"),
            ('Zero Copy', 'Zero Copy in', "#FF0000")  # Red color code added here
        ], f"{modes} {SF} Data I/O comparison", 'data_sizes.png')
    ]
    
    for loader, config, title, outfile in breakdowns:
        results = loader(prefix, SF, qid, modes)
        plot_stacked_bar(
            queries, results, config,
            title, os.path.join('experiments/figures', outfile), modes
        )
        # mode1_cpu_total = results["in_mem"]["CPU in"] + results["in_mem"]["CPU out"]
        # mode2_cpu_total = results["out_mem"]["CPU in"] + results["out_mem"]["CPU out"]
        # print('avg ratio', (mode1_cpu_total / mode2_cpu_total).mean())
        # print('avg ratio', (mode1_cpu_total / mode2_cpu_total).min())

if __name__ == '__main__':
    main()