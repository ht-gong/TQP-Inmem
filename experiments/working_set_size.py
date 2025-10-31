#!/usr/bin/env python3
"""
Script to extract numeric values following specific keywords from a predefined list of log files, plot them, and generate a summary bar chart of all working set consumptions.

Usage:
    python extract_and_plot_logs.py \
        --input-dir logs/ [--prefix size_] [--suffix .log] [--output-dir plots] [--bar-plot]

Modify the INDICES list below to match your log file numbering.
By default, it processes files named PREFIX{idx}SUFFIX for each idx in INDICES.

Flags:
  --bar-plot    Generate a bar plot of all 'working set' values across all log files.

For each file, it extracts integers (4+ digits) after 'working set', 'input size', or 'output size',
then generates a line+scatter plot per file, and optionally a bar chart of all working sets.
"""

import re
import os
import argparse
import matplotlib.pyplot as plt

# List of indices to process (edit this list as needed)
INDICES = list(range(1, 23))  # e.g., [1, 2, ..., 22]

# Marker styles per keyword
MARKER_MAP = {
    'working set': 'o',    # circle
    'input size': 's',     # square
    'output size': '^',    # triangle
}

# Bytes to GB conversion
GB = 1024 ** 3


def extract_tagged_values(line):
    """
    Extract (keyword, value) tuples where the keyword is one of our targets
    and the value is an integer with 4 or more digits following it.
    Values are converted from bytes to GB.
    """
    pattern = re.compile(r"\b(working set|input size|output size)\b.*?(\d{1,})",
                         flags=re.IGNORECASE)
    return [(m.group(1).lower(), int(m.group(2)) / GB) for m in pattern.finditer(line)]

def extract_join_values(line):
    """
    Extract (keyword, value) tuples where the keyword is one of our targets
    and the value is an integer with 4 or more digits following it.
    Values are converted from bytes to GB.
    """
    pattern = re.compile(r"(input size)\b.*?(partition).*?(\d{1,})",
                         flags=re.IGNORECASE)
    return [(m.group(1).lower(), int(m.group(3)) / 4) for m in pattern.finditer(line)]


def plot_for_file(filepath, output_dir=None):
    tagged = []
    with open(filepath, 'r') as f:
        for line in f:
            tagged.extend(extract_tagged_values(line))
    if not tagged:
        print(f"No matching values found in {os.path.basename(filepath)}")
        return

    x = list(range(len(tagged)))
    y = [val for (_, val) in tagged]
    keys = [key for (key, _) in tagged]

    plt.figure()
    # Grey trend line
    plt.plot(x, y, linestyle='-', color='gray', alpha=0.5)

    # Scatter per category
    for keyword, marker in MARKER_MAP.items():
        xs = [i for i, k in enumerate(keys) if k == keyword]
        ys = [v for (k, v) in tagged if k == keyword]
        if xs:
            plt.scatter(xs, ys, marker=marker, label=keyword)

    # Example horizontal line (customize as needed)
    plt.axhline(y=6, linestyle=':', color='blue', label='Working Set Capacity (GB)')

    plt.xlabel('Occurrence Index')
    plt.ylabel('GB')
    plt.title('Query ' + os.path.basename(filepath).split('_')[1].split('.')[0])
    plt.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filepath))[0]
        outpath = os.path.join(output_dir, f"{base}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"Saved plot for {filepath} -> {outpath}")
    else:
        plt.show()


def plot_hash_partition_bar(input_dir, prefix, suffix, output_dir=None):
    ws_list = []  # list of (label, value)

    for idx in INDICES:
        filename = f"{prefix}{idx}{suffix}"
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r') as f:
            for line in f:
                for key, val in extract_join_values(line):
                    ws_list.append(val)

    if not ws_list:
        print("No working set values found for any files.")
        return

    sum_list = []
    # assert len(ws_list) // 2 == 0
    for i in range(len(ws_list) // 2):
        sum_list.append((ws_list[i * 2] + ws_list[i * 2 + 1]) * 10)
    ws_list_sorted = sorted(sum_list, key=lambda x: x, reverse=True)
    values = ws_list_sorted

    indices = range(len(values))
    plt.figure()
    plt.bar(indices, values)
    plt.xlabel('Operation Index')
    plt.ylabel('Hash Join (Rows)')
    plt.title('Hash Join Scale')
    # plt.xticks(indices, labels, rotation='vertical', fontsize=6)
    ax = plt.gca()
    y0 = 2e9
    ax.axhline(y=y0, linestyle=':', color='blue')
    ax.text(
    50, y0, '2 Cols Fit',
    ha='center', va='bottom'
    )

    # y0 = 8e9
    # ax.axhline(y=y0, linestyle=':', color='red')
    # ax.text(
    # 50, y0, '1 Col Fits',
    # ha='center', va='bottom'
    # )
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, 'hash_join_bar_10x.png')
        plt.savefig(outpath)
        plt.close()
        print(f"Saved working set bar plot -> {outpath}")
    else:
        plt.show()

def plot_working_set_bar(input_dir, prefix, suffix, output_dir=None):
    """
    Collect all 'working set' values from each log file and plot a bar chart,
    sorted from highest to lowest consumption.
    """
    ws_list = []  # list of (label, value)

    for idx in INDICES:
        filename = f"{prefix}{idx}{suffix}"
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r') as f:
            for line in f:
                for key, val in extract_tagged_values(line):
                    if key == 'working set':
                        ws_list.append((str(idx), val))

    if not ws_list:
        print("No working set values found for any files.")
        return

    ws_list_sorted = sorted(ws_list, key=lambda x: x[1], reverse=True)
    labels, values = zip(*ws_list_sorted)

    indices = range(len(values))
    plt.figure()
    plt.bar(indices, values)
    plt.xlabel('Operation Index')
    plt.ylabel('Working Set (GB)')
    plt.title('All Working Set Consumptions')
    # plt.xticks(indices, labels, rotation='vertical', fontsize=6)
    ax = plt.gca()
    y0 = 6
    ax.axhline(y=y0, linestyle=':', color='blue')
    ax.text(
    300, y0, 'GPU Working Set Capacity',
    ha='center', va='bottom'
    )
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, 'working_set_bar.png')
        plt.savefig(outpath)
        plt.close()
        print(f"Saved working set bar plot -> {outpath}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot tagged values from multiple log files using INDICES list')
    parser.add_argument('--input-dir', '-d', default='.',
                        help='Directory where log files reside (default: current directory)')
    parser.add_argument('--prefix', '-p', default='size_',
                        help='Filename prefix (default: size_)')
    parser.add_argument('--suffix', '-s', default='.log',
                        help='Filename suffix (default: .log)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory to save plots (interactive display if omitted)')
    args = parser.parse_args()

    plot_hash_partition_bar(args.input_dir, args.prefix, args.suffix, args.output_dir)
    # plot_working_set_bar(args.input_dir, args.prefix, args.suffix, args.output_dir)
    # for idx in INDICES:
    #     filename = f"{args.prefix}{idx}{args.suffix}"
    #     filepath = os.path.join(args.input_dir, filename)
    #     if not os.path.isfile(filepath):
    #         print(f"Warning: {filename} not found in {args.input_dir}, skipping.")
    #         continue
    #     plot_for_file(filepath, args.output_dir)

if __name__ == '__main__':
    main()
