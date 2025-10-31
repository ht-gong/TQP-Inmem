#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_json_total_breakdown(dat, SF, qid, modes, times):
    results = {}
    for mode in modes:
        gpu_tots, io_tots, other_tots, gpu_non_pipe = ([] for _ in range(4))
        for Q in qid:
            path = f"experiments/results/{dat}/{mode}/{SF}_{Q}.json"
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}", file=sys.stderr)
                gpu_tots.append(io_tots.append(other_tots.append(0) or 0) or 0)
                times[mode][Q] = 0
                continue
            tot = next(v['total'] for k, v in data.items() if re.match(r"Query", k))
            ops = {k: v for k, v in data.items() if re.match(r"^[a-z]{2,} \d+$", k)}
            for op in list(ops):
                if 'scan' in op or 'output' in op:
                    tot -= ops[op]['total']
            # pinned = sum(v['total'] for k, v in data.items() if re.search(r'pin', k, re.IGNORECASE) and 'scan' not in k)
            gpu = sum(v['total'] for k, v in data.items() if re.match(r'.*apply op.*', k))
            # gpu_unpipe = sum(v['total'] for k, v in data.items() if re.match(r'.*blocking apply.*', k))
            io_ = sum(v['total'] for k, v in data.items() if re.match(r'.*transfer.*', k))
            gpu_tots.append(gpu); io_tots.append(io_); other_tots.append(tot - gpu  - io_)
            times[mode][Q] = tot
        # results[mode] = {'GPU unpiped': np.array(gpu_non_pipe), 'GPU':np.array(gpu_tots),'IO':np.array(io_tots),'other':np.array(other_tots)}
        results[mode] = {'GPU':np.array(gpu_tots),'IO':np.array(io_tots),'other':np.array(other_tots)}
    return results


def load_json_operator_breakdown(dat, SF, qid, modes):
    results = {}
    for mode in modes:
        f_t, f_c, f_cpu = [], [], []
        a_t, a_c, a_cpu = [], [], []
        j_t, j_c, j_cpu = [], [], []
        s_t = []
        o_t = []
        for Q in qid:
            path = f"experiments/results/{dat}/{mode}/{SF}_{Q}.json"
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}", file=sys.stderr)
                f_t.append(f_c.append(f_cpu.append(a_t.append(a_c.append(a_cpu.append(j_t.append(j_c.append(j_cpu.append(s_t.append(o_t.append(0) or 0) or 0) or 0) or 0) or 0) or 0) or 0) or 0)))
                continue
            tot = next(v['total'] for k, v in data.items() if re.match(r"Query", k))
            ops = {k: v for k, v in data.items() if re.match(r"^[a-z]{2,} \d+$", k)}
            for op in list(ops):
                if 'scan' in op:
                    tot -= ops[op]['total']
            filter_compute   = sum(v['total'] for k, v in data.items() if re.match(r'^filter\s+\d+.*apply op.*$', k))
            filter_transfer  = sum(v['total'] for k, v in data.items() if re.match(r'^filter\s+\d+.*transfer.*$', k))
            filter_all       = sum(v['total'] for k, v in data.items() if re.match(r'^filter\s+\d+$', k))
            agg_compute      = sum(v['total'] for k, v in data.items() if re.match(r'^aggregation\s+\d+.*apply op.*$', k) or re.match(r'.*aggregation.*prologue.*', k))
            agg_transfer     = sum(v['total'] for k, v in data.items() if re.match(r'^aggregation\s+\d+.*transfer.*$', k))
            agg_all          = sum(v['total'] for k, v in data.items() if re.match(r'^aggregation\s+\d+$', k))
            join_compute     = sum(v['total'] for k, v in data.items() if re.match(r'^join\s+\d+.*apply op.*$', k))
            join_transfer    = sum(v['total'] for k, v in data.items() if re.match(r'^join\s+\d+.*transfer.*$', k))
            join_all         = sum(v['total'] for k, v in data.items() if re.match(r'^join\s+\d+$', k))
            sort_compute     = sum(v['total'] for k, v in data.items() if re.match(r'^sort\s+\d+$', k))
            f_t.append(filter_transfer); f_c.append(filter_compute); f_cpu.append(filter_all - filter_transfer - filter_compute)
            a_t.append(agg_transfer); a_c.append(agg_compute); a_cpu.append(agg_all - agg_transfer - agg_compute)
            j_t.append(join_transfer); j_c.append(join_compute); j_cpu.append(join_all - join_transfer - join_compute)
            s_t.append(sort_compute)
            o_t.append(tot - filter_all - agg_all - join_all - sort_compute)
        results[mode] = {
            'filter I/O': np.array(f_t), 'filter compute': np.array(f_c), 'filter others': np.array(f_cpu),
            'agg I/O':    np.array(a_t), 'agg compute':    np.array(a_c), 'agg others':    np.array(a_cpu),
            'join I/O':   np.array(j_t), 'join compute':   np.array(j_c), 'join others':   np.array(j_cpu),
            'sort':       np.array(s_t), 'other':          np.array(o_t)
        }
    return results


def load_json_sync_breakdown(dat, SF, qid, modes):
    results = {}
    for mode in modes:
        cat_p, rearr_p, part_p, mask_p = [], [], [], []
        cg, io_p, co, o_t = [], [], [], []
        for Q in qid:
            path = f"experiments/results/{dat}/{mode}/{SF}_{Q}.json"
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}", file=sys.stderr)
                cat_p.append(rearr_p.append(part_p.append(mask_p.append(cg.append(io_p.append(co.append(o_t.append(0) or 0) or 0) or 0) or 0) or 0) or 0) or 0)
                continue
            tot = next(v['total'] for k, v in data.items() if re.match(r"Query", k))
            ops = {k: v for k, v in data.items() if re.match(r"^[a-z]{2,} \d+$", k)}
            for op in list(ops):
                if 'scan' in op:
                    tot -= ops[op]['total']
            cat = sum(v['total'] for k, v in data.items() if re.match(r'.*cat list pipeline$', k))
            rearr = sum(v['total'] for k, v in data.items() if re.match(r'.*rearrange pipe$', k))
            part  = sum(v['total'] for k, v in data.items() if re.match(r'.*partition pipe$', k))
            mask  = sum(v['total'] for k, v in data.items() if re.match(r'.*mask pipe$', k))
            comp_g= sum(v['total'] for k, v in data.items() if re.match(r'.*(filter|join|aggregation) pipe.*apply op$', k))
            comp_i= sum(v['total'] for k, v in data.items() if re.match(r'.*(filter|join|aggregation) pipe.*transfer.*$', k))
            comp_t= sum(v['total'] for k, v in data.items() if re.match(r'.*(filter|join|aggregation) pipe$', k))
            cat_p.append(cat); rearr_p.append(rearr); part_p.append(part); mask_p.append(mask)
            cg.append(comp_g); io_p.append(comp_i); co.append(comp_t - comp_g - comp_i)
            o_t.append(tot - cat - rearr - part - mask - comp_t)
        results[mode] = {
            'cat pipeline':     np.array(cat_p),
            'rearrange pipeline':np.array(rearr_p),
            'partition pipeline':np.array(part_p),
            'mask pipeline':    np.array(mask_p),
            'compute GPU':      np.array(cg),
            'compute I/O':      np.array(io_p),
            'compute other':    np.array(co),
            'other':            np.array(o_t)
        }
    return results


def load_baseline(csv_path, qid, times, baseline_name):
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['query'], df['runtime_s']))
    baseline = np.array([mapping.get(Q, 0) for Q in qid])
    times[baseline_name] = baseline

def load_baseline_ms(csv_path, qid, times, baseline_name):
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['query'], df['time/milliseconds']))
    baseline = np.array([mapping.get(str(Q), 0) / 1000 for Q in qid])
    times[baseline_name] = baseline

def print_averages(times, modes, baseline_times, qid, *, eps=0.0):
    """
    Uses geometric mean for 'avg_ratio'.
    - times[mode][Q]: runtime of mode on query Q
    - baseline_times[k][i]: baseline runtime aligned with qid[i]
    - If any ratio <= 0, set eps>0 (e.g., 1e-12) to stabilize: r = max(r, eps)
    """
    print(qid)
    for mode in modes:
        for k in baseline_times:
            # per-query ratios, aligned by qid order
            ratios = [
                (baseline_times[k][i] / times[mode][Q])
                for i, Q in enumerate(qid)
            ]

            if eps > 0.0:
                ratios = [max(r, eps) for r in ratios]

            # geometric mean of ratios
            # (requires all ratios > 0; will ValueError on non-positive if eps==0)
            if any(r <= 0 for r in ratios):
                raise ValueError("Geometric mean requires positive ratios; set eps>0 to clamp.")

            gm_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

            # total (harmonic of sums as you had)
            tot_ratio = (sum(baseline_times[k][i] for i, _ in enumerate(qid))
                         / sum(times[mode][Q] for Q in qid))

            print(f"Average (geo) for {mode} vs {k}: {gm_ratio:.6g}, Tot: {tot_ratio:.6g}")


def plot_stacked_bar(
        queries,
        results,
        baselines,                 # Dict[str, np.ndarray]
        config,                    # iterable of (label, key, color) for stacked sections
        title,
        outfile,
        modes,                     # list of experimental modes to stack
        legend_loc="upper right"):
    """
    Draw a grouped stacked-bar chart.

    Parameters
    ----------
    queries   : list[Any]               # x-axis tick labels
    results   : Dict[str, Dict[str,np.ndarray]]
                # results[mode][metric] = 1-D array (len = len(queries))
    baselines : Dict[str, np.ndarray]   # each array len == len(queries)
    config    : iterable[(label, key, color)]
    modes     : list[str]               # stacked modes to plot
    """

    all_modes = modes + list(baselines.keys())      # order on the x‑axis
    num_bars  = len(all_modes)
    bar_w     = 0.8 / num_bars                      # keep total group width ≈ 0.8
    offsets   = {m: bar_w * (i - (num_bars - 1) / 2)
                 for i, m in enumerate(all_modes)}

    x = np.arange(len(queries))

    plt.figure(figsize=(14, 6))

    # Stacked experimental modes
    for mode in modes:
        bottom = np.zeros(len(queries), dtype=float)
        for lbl, key, color in config:
            vals = results[mode][key]
            plt.bar(x + offsets[mode],
                    vals,
                    bar_w,
                    bottom=bottom,
                    label=lbl if mode == modes[0] else None,  # one legend entry per stack section
                    color=color,
                    edgecolor="black",
                    linewidth=0.2)
            bottom += vals

    # Baseline bars (single‑segment)
    baseline_colors = baseline_colors = [
            "#8b008b",  # dark magenta
            "#2e0854",  # very‑dark indigo
            "#4b0082",  # indigo
            "#5e3c99",  # muted dark violet
            "#6a0dad",  # royal purple
            "#9932cc",  # dark orchid
            "#8a2be2",  # blue‑violet
            "#551a8b",  # purple heart
            "#3f0071",  # midnight plum
        ]
    for i, (name, vals) in enumerate(baselines.items()):
        plt.bar(x + offsets[name],
                vals,
                bar_w,
                label=name,
                color=baseline_colors[i % len(baseline_colors)],
                edgecolor="black",
                linewidth=0.2)

    # ------------- cosmetics --------------------------------------------------
    plt.xticks(x, queries, rotation=45)
    plt.ylabel("Total Time (s)")
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()


def main():
    # qid = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 19, 20, 22]
    qid = list(range(1, 23))
    dat   = "2025-09-07"
    SF    = "SF100"
    modes = ['hashed_unique']
    times = {mode: {} for mode in modes}

    baseline_times = {}
    # load_baseline('experiments/results/2025-08-13/SF100_no_overlap_io_times.csv', qid, baseline_times, 'Only I/O Lower Bound')
    load_baseline('experiments/results/duckdb_sf100.csv', qid, baseline_times, 'DuckDB')

    # load_baseline('experiments/results/2025-09-16/SF300_no_overlap_io_times.csv', qid, baseline_times, 'Only I/O Lower Bound')
    # load_baseline('experiments/results/duckdb_sf300.csv', qid, baseline_times, 'DuckDB')
    # load_baseline_ms('experiments/results/sparktime_sf100.csv', qid, baseline_times, 'Spark')
    queries = [f"Q{q}" for q in qid]

    breakdowns = [
        (load_json_total_breakdown,  [
            # ('GPU (blocking)', 'GPU unpiped', '#56B4E9'), ('GPU', 'GPU', '#1f77b4'), ('IO (non-overlapped)', 'IO', '#2ca02c'),
            ('GPU', 'GPU', '#1f77b4'), ('IO (non-overlapped)', 'IO', '#2ca02c'),
            ('Others', 'other', '#7f7f7f')
        ], f"{modes} Total Query Breakdown", 'query_overheads_with_baseline.png'),
        (load_json_operator_breakdown, [
            ('Filter I/O (non-overlapped)','filter I/O','#FF9896'), ('Filter Compute','filter compute','#D62728'), ('Filter Others','filter others','#FFCDCD'),
            ('Agg I/O (non-overlapped)','agg I/O','#98DF8A'), ('Agg Compute','agg compute','#2CA02C'), ('Agg Others','agg others','#CDECC6'),
            ('Join I/O (non-overlapped)','join I/O','#AEC7E8'), ('Join Compute','join compute','#1F77B4'), ('Join Others','join others','#C6DDFA'),
            ('Sort','sort','#FF7F0E'), ('Other','other','#7F7F7F')
        ], f"{modes} Operator Breakdown", 'query_overheads_with_agg_breakdown.png'),
        (load_json_sync_breakdown, [
            ('Cat','cat pipeline','#56B4E9'), ('Rearrange','rearrange pipeline','#E69F00'),
            ('Partition','partition pipeline','#D55E00'), ('Mask','mask pipeline','#009E73'),
            ('Core GPU','compute GPU','#0C2340'), ('Core I/O (non-overlapped)','compute I/O','#255DAD'), ('Core Other','compute other','#6497F0'),
            ('Other','other','#999999')
        ], f"{modes} Pipeline Breakdown", 'query_overheads_with_pipe_breakdown.png')
    ]

    for loader, config, title, outfile in breakdowns:
        if loader is load_json_total_breakdown:
            results = loader(dat, SF, qid, modes, times)
        else:
            results = loader(dat, SF, qid, modes)
        plot_stacked_bar(
            queries, results, baseline_times, config,
            title, os.path.join('experiments/figures', outfile), modes
        )
    print_averages(times, modes, baseline_times, qid)


if __name__ == '__main__':
    main()