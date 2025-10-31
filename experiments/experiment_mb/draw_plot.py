import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import blended_transform_factory
# CSV data provided by user
df = pd.read_csv('/work1/talati/haotiang/TQP-Vortex/experiments/experiment_mb/sorting_results.csv')

def breakdown():
    global df
    df = df[df['Size_GB'].isin([0.01, 1, 3, 20, 50])]
    vt = df['Vortex transfer']
    vc = df['Vortex compute']
    v = df['Vortex']
    nvt = df['No Vortex transfer']
    nvc = df['No Vortex compute']
    nv = df['No Vortex']

    vt_pct = vt / v * 100
    vc_pct = vc / v * 100
    ov_pct = 100 - (vt_pct + vc_pct)

    nvt_pct = nvt / nv * 100
    nvc_pct = nvc / nv * 100
    on_pct = 100 - (nvt_pct + nvc_pct)

    # Plot
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, vt_pct, width, label='Vortex transfer')
    ax.bar(x - width/2, vc_pct, width, bottom=vt_pct, label='Vortex compute')
    ax.bar(x - width/2, ov_pct, width, bottom=vt_pct+vc_pct, label='Vortex other')

    ax.bar(x + width/2, nvt_pct, width, label='No Vortex transfer')
    ax.bar(x + width/2, nvc_pct, width, bottom=nvt_pct, label='No Vortex compute')
    ax.bar(x + width/2, on_pct, width, bottom=nvt_pct+nvc_pct, label='No Vortex other')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Size_GB'])
    ax.set_xlabel('Size (GB)')
    ax.set_ylabel('Percentage of Total Time (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('experiments/experiment_mb/sort_runtimes_breakdown.png')
    plt.show()

def all():
    plt.figure()

    # Plot GPU and CPU
    plt.plot(df['Size_GB'], df['GPU'], label='GPU', marker='o')
    plt.plot(df['Size_GB'], df['CPU'], label='CPU',marker='o')

    # Plot Vortex and its components with matching color
    line_vortex, = plt.plot(df['Size_GB'], df['Vortex'], label='Vortex', marker='o')

    # Plot No Vortex and its components with matching color
    line_no, = plt.plot(df['Size_GB'], df['No Vortex'], label='No Vortex', marker='o')

    # Plot DuckDB
    plt.plot(df['Size_GB'], df['DuckDB'], label='DuckDB', marker='o')

    plt.xlabel('Size (GB)')
    plt.ylabel('Runtime (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    output_path = 'experiments/experiment_mb/sort_runtimes.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_throughput_5_and_30gb():
    # 1) Filter and compute throughputs
    methods = ['GPU', 'CPU', 'Vortex', 'No Vortex', 'DuckDB']
    sizes = [1, 30]
    throughputs = {}
    for sz in sizes:
        row = df[df['Size_GB'] == sz]
        if row.empty:
            raise ValueError(f"No data for Size_GB == {sz}")
        throughputs[sz] = [(sz / 4.0) / row[m].iloc[0] for m in methods]

    # 2) Set up bar positions
    x = np.arange(len(methods))
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, throughputs[sizes[0]], width, label=f"{sizes[0]} GB")
    ax.bar(x + width/2, throughputs[sizes[1]], width, label=f"{sizes[1]} GB")

    # 3) Labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlabel("Method")
    ax.set_ylabel("Throughput (B Tuples/s)")
    ax.set_title("Sort Throughput @ 1 GB vs 30 GB")
    ax.legend(title="Size")

    plt.tight_layout()

    # 4) Save & show
    out = 'experiments/experiment_mb/sort_throughput_1_and_30gb.png'
    plt.savefig(out)
    plt.show()
    print(f"Saved throughput plot to {out}")

if __name__ == "__main__":
    all()
    plot_throughput_5_and_30gb()
    # breakdown()