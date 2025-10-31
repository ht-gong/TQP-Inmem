import argparse
import json, os
import re
import sys
import matplotlib.pyplot as plt

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Load a JSON file and extract top-level keys matching a regex."
    )
    parser.add_argument(
        "json_path",
        help="Path to the JSON file to read"
    )
    # python ./experiments/parse_log.py -json_path=./experiments/May17/q1.log -figure_path ./experiments/figures/q1.png -query_name q1
    parser.add_argument(
        "figure_path",
        help="Path to the figure folder",
        default="/work1/talati/zhaoyah/TQP-Vortex/experiments/figures"
    )
    parser.add_argument(
        "query_name",
        help="Executed_query",
        default="Q1"
    )
    
    args = parser.parse_args()

    # 2. Load JSON
    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    query_regex = re.compile("Query")
    tot_time = [v for k, v in data.items() if query_regex.match(k)][0]['total']
    
    operators_regex = re.compile("^[a-z]{2,} \d+$")
    matches = {k: v for k, v in data.items() if operators_regex.match(k)}
    for key, value in matches.items():
        print(key, value['total'])

    # 1. Sort operators by total descending
    sorted_ops = sorted(
        matches.items(),
        key=lambda kv: kv[1]['total'],
        reverse=True
    )

    # 2. Split into top 10 and the rest
    top10   = sorted_ops[:6]
    others  = sorted_ops[6:]

    # 3. Build labels/sizes for top 10
    labels = [k for k, _ in top10]
    sizes  = [v['total'] / tot_time * 100 for _, v in top10]

    # 4. Aggregate the rest as “Other”
    if others:
        other_total = sum(v['total'] for _, v in others)
        labels.append("Other")
        sizes.append(other_total / tot_time * 100)

    # 7. Draw pie chart
    plt.figure()
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Operator Time as % of Total Query Time")
    plt.axis('equal')
    plt.tight_layout()
    output_path = os.path.join(args.figure_path,f"{args.query_name}_operator_time_pie.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved pie chart to ./{output_path}")

    gpu_compute_regex = re.compile(r'^.*apply op.*$')
    gpu_tot = sum([v['total'] for k, v in data.items() if gpu_compute_regex.match(k)])

    cpu_compute_regex = re.compile(r'^.*CPU.*$')
    cpu_tot = sum([v['total'] for k, v in data.items() if cpu_compute_regex.match(k)])

    gpu_IO_regex = re.compile(r'^.*transfer.*$')
    io_tot = sum([v['total'] for k, v in data.items() if gpu_IO_regex.match(k)])    

    labels=['GPU Compute', 'CPU Compute', 'CPU-GPU IO']
    
    plt.figure()
    patches, texts, autotexts = plt.pie(
        [gpu_tot, cpu_tot, io_tot],
        autopct='%1.1f%%',    # show percent to one decimal
        startangle=90         # rotate so first slice starts at 12 o'clock
    )
    plt.title("Operator Time as % of Total Query Time")
    plt.axis('equal')        # keep pie chart circular
    plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 0.5))  # 图例放右侧居中
    plt.tight_layout()
    output_path = os.path.join(args.figure_path, f"{args.query_name}_device_time_pie.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved pie chart to ./{output_path}")

if __name__ == "__main__":
    main()
