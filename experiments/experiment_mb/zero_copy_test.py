import argparse
import statistics as stats
import time, sys, os

# Make sure your built extension is importable
sys.path.insert(0, "/home/jovyan/workspace/TQP-Vortex/IO/build")
import cuda_zero_copy_reader
import torch
import argparse
import statistics as stats
import time, sys, os
from collections import defaultdict

import matplotlib.pyplot as plt


# def fmt_ns(ns: int) -> str:
#     return f"{ns/1e6:.3f} ms"


# def bench_rearrange(n_bytes: int, mask_probs,  group_sizes, iters: int, seed: int = 123456):
#     CHUNK_SIZE = 2 * 1024 * 1024 ** 2
#     assert torch.cuda.is_available(), "CUDA not available"
#     device_cpu = torch.device("cpu")
#     device_gpu = torch.device("cuda:0")
#     # Use int32 (4 bytes/elem)
#     elem_size = 4
#     n = n_bytes // elem_size // group_size
#     if n * elem_size * group_size != n_bytes:
#         print(f"Note: rounding down to {n*elem_size*group_size} bytes ({n} groups).")   

#     # Fixed data on CPU (pinned); same across all p to isolate mask effect
#     g_cpu = torch.Generator(device=device_cpu).manual_seed(seed)
#     x_cpu = torch.randint(0, 256, (n, group_size), device=device_cpu, dtype=torch.int32, generator=g_cpu).contiguous().pin_memory()
#     # print(x_cpu)
#     # Generate one uniform(0,1) vector on GPU once, reuse for all p as (u < p).
#     g_gpu = torch.Generator(device=device_gpu).manual_seed(seed)
#     u = torch.randperm(n, device=device_gpu, generator=g_gpu)
#     p0 = mask_probs[0]
#     arrange0 = u[:int(n * p0)]
#     # print(arrange0)

#     results = []
#     for p in mask_probs:
#         arrange0 = u[:int(n * p)]

#         cpu_times = []
#         gpu_times = []

#         # Short per-p warmup (keeps fairness if p strongly changes work)
#         for _ in range(2):
#             out = torch.empty_like(x_cpu)[:int(n*p)].to('cuda')
#             cuda_zero_copy_reader.zero_copy_rearrange(x_cpu, arrange0, out)
#         for _ in range(2):
#             xg = x_cpu.to(device_gpu)
#             _ = xg[arrange0]
#             torch.cuda.synchronize()

#         # Timed loops
        
#         res1 = torch.empty_like(x_cpu)[:int(n*p)].to('cuda')
#         for _ in range(iters):
#             # res1 = torch.empty_like(x_cpu)[:ints(n*p)].to('cuda')
#             t0 = time.perf_counter_ns()
#             cuda_zero_copy_reader.zero_copy_rearrange(x_cpu, arrange0, res1)
#             # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#             torch.cuda.synchronize()
#             t1 = time.perf_counter_ns()
#             cpu_times.append(t1 - t0)

#         res2 = None
#         for _ in range(iters):
#             t0 = time.perf_counter_ns()
#             xg = x_cpu.to(device_gpu)            # H2D copy
#             res2 = xg[arrange0]
#             torch.cuda.synchronize()
#             t1 = time.perf_counter_ns()
#             gpu_times.append(t1 - t0)
#         assert torch.equal(res1, res2)

#         med_cpu_ms = stats.median(cpu_times)/1e6
#         med_gpu_ms = stats.median(gpu_times)/1e6
#         print(f"p={p:.6f}  median_cpu={med_cpu_ms:.3f} ms   median_gpu={med_gpu_ms:.3f} ms")
#         results.append((float(p), float(med_cpu_ms), float(med_gpu_ms)))    

#     return results
    
# def bench_mask(n_bytes: int, mask_probs, group_sizes, iters: int, seed: int = 123456):
#     """
#     For each p in mask_probs, measure median time (ms) for:
#       - CPU zero-copy masked read (then compact on GPU)
#       - GPU copy of data then masked select on GPU
#     Returns: list of (p, median_cpu_ms, median_gpu_ms)
#     """
#     assert torch.cuda.is_available(), "CUDA not available"
#     device_cpu = torch.device("cpu")
#     device_gpu = torch.device("cuda:0")


#     # Use float32 (4 bytes/elem)
#     elem_size = 4
#     n = n_bytes // elem_size // group_size
#     if n * elem_size * group_size != n_bytes:
#         print(f"Note: rounding down to {n*elem_size*group_size} bytes ({n} groups).")   


#     # Fixed data on CPU (pinned); same across all p to isolate mask effect
#     g_cpu = torch.Generator(device=device_cpu).manual_seed(seed)
#     x_cpu = torch.randint(0, 256, (n, group_size), device=device_cpu, dtype=torch.int32, generator=g_cpu).contiguous().pin_memory()

#     # Generate one uniform(0,1) vector on GPU once, reuse for all p as (u < p).
#     g_gpu = torch.Generator(device=device_gpu).manual_seed(seed)
#     u = torch.rand(n, device=device_gpu, generator=g_gpu)

#     out = torch.empty_like(x_cpu).to('cuda')
#     p0 = mask_probs[0]
#     mask0 = (u < p0)

#     # Sanity-check equivalence once
#     cuda_zero_copy_reader.zero_copy_mask(x_cpu, mask0, out)
#     out_1 = out[mask0]
#     xg = x_cpu.to(device_gpu)
#     out_2 = xg[mask0]
#     torch.cuda.synchronize()
#     if not torch.equal(out_1, out_2):
#         raise RuntimeError("Sanity check failed: zero-copy path != copy+mask")

#     results = []
#     for p in mask_probs:
#         mask = (u < p)

#         cpu_times = []
#         gpu_times = []

#         # Short per-p warmup (keeps fairness if p strongly changes work)
#         for _ in range(2):
#             out = torch.empty_like(x_cpu).to('cuda')
#             cuda_zero_copy_reader.zero_copy_mask(x_cpu, mask, out)
#             _ = out[mask]
#         for _ in range(2):
#             xg = x_cpu.to(device_gpu)
#             _ = xg[mask]
#             torch.cuda.synchronize()

#         # Timed loops
#         for _ in range(iters):
#             out = torch.empty_like(x_cpu).to('cuda')
#             t0 = time.perf_counter_ns()
#             cuda_zero_copy_reader.zero_copy_mask(x_cpu, mask, out)
#             _ = out[mask]
#             torch.cuda.synchronize()
#             t1 = time.perf_counter_ns()
#             cpu_times.append(t1 - t0)

#         for _ in range(iters):
#             t0 = time.perf_counter_ns()
#             xg = x_cpu.to(device_gpu)            # H2D copy
#             _ = xg[mask]
#             torch.cuda.synchronize()
#             t1 = time.perf_counter_ns()
#             gpu_times.append(t1 - t0)

#         med_cpu_ms = stats.median(cpu_times)/1e6
#         med_gpu_ms = stats.median(gpu_times)/1e6
#         print(f"p={p:.6f}  median_cpu={med_cpu_ms:.3f} ms   median_gpu={med_gpu_ms:.3f} ms")
#         results.append((float(p), float(med_cpu_ms), float(med_gpu_ms)))    

#     return results
    


# def main():
#     parser = argparse.ArgumentParser(description="Plot median latency vs p for zero-copy vs copy+mask")
#     parser.add_argument("--bytes", type=int, required=True,
#                         help="Total tensor size in bytes (float32, 4B/elem)")
#     parser.add_argument("--iters", type=int, default=20,
#                         help="Iterations per p (default 20)")
#     parser.add_argument("--seed", type=int, default=123456,
#                         help="RNG seed (default 123456)")
#     parser.add_argument("--ps", type=float, nargs="+",
#                         default=[1, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
#                         help="Mask probabilities to test")
#     parser.add_argument("--gs", type=float, nargs="+",
#                         default=[1, 2, 3, 4, 5, 10, 15, 20, 25],
#                         help="Group_sizes to test")
#     parser.add_argument("--out-prefix", type=str, default="median_vs_p",
#                         help="Prefix for CSV/PNG outputs")
#     args = parser.parse_args()

#     ps = list(args.ps)
#     gs = list(args.gs)
#     results = bench_rearrange(args.bytes, ps, args.iters, gs, seed=args.seed)

#     print(results)
#     # Save CSV (no pandas needed)
#     csv_path = f"{args.out_prefix}.csv"
#     with open(csv_path, "w", encoding="utf-8") as f:
#         f.write("p,median_cpu_ms,median_gpu_ms\\n")
#         for p, m_cpu, m_gpu in results:
#             f.write(f"{p},{m_cpu},{m_gpu}\\n")
#     print(f"Saved CSV -> {csv_path}")

#     # Plot (single figure, log-x)
#     # Sort by p descending just for a cleaner line
#     results_sorted = sorted(results, key=lambda r: r[0], reverse=True)
#     ps_sorted = [r[0] for r in results_sorted]
#     cpu_sorted = [r[1] for r in results_sorted]
#     gpu_sorted = [r[2] for r in results_sorted]

#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(ps_sorted, cpu_sorted, marker="o", label="CPU zero-copy path (median)")
#     ax.plot(ps_sorted, gpu_sorted, marker="o", label="GPU copy+mask (median)")
#     ax.set_xscale("log")
#     ax.set_xlabel("mask probability p")
#     ax.set_ylabel("median latency (ms)")
#     ax.set_title(f"Latency of PCIE transfer (Slab copy vs. Zero-copy)  (GB={args.bytes // (1024**3)}, iters={args.iters})")
#     ax.grid(True, which="both", linestyle="--", alpha=0.5)
#     ax.legend()
#     fig.tight_layout()
#     png_path = f"{args.out_prefix}.png"
#     fig.savefig(png_path, dpi=160)
#     print(f"Saved plot -> {png_path}")
#     # Make sure your built extension is importable

import matplotlib.pyplot as plt


def fmt_ns(ns: int) -> str:
    return f"{ns/1e6:.3f} ms"


def run_benchmark(mode: str, n_bytes: int, mask_probs, group_sizes, iters: int, seed: int = 123456):
    """
    For each group_size and p, measure median time (ms) for:
      - CPU zero-copy path
      - GPU copy of data then operation on GPU
    Returns: list of (group_size, p, median_cpu_ms, median_gpu_ms)
    """
    assert torch.cuda.is_available(), "CUDA not available"
    assert mode in ["mask", "rearrange"], "mode must be 'mask' or 'rearrange'"
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda:0")
    elem_size = 4  # Use int32 (4 bytes/elem)

    results = []
    for group_size in group_sizes:
        print(f"\n--- Benchmarking group_size = {group_size} ---")
        n = n_bytes // elem_size // group_size
        if n * elem_size * group_size != n_bytes:
            print(f"Note: rounding down to {n*elem_size*group_size} bytes ({n} groups of size {group_size}).")

        # Fixed data on CPU (pinned); same across all p to isolate mask effect
        g_cpu = torch.Generator(device=device_cpu).manual_seed(seed)
        x_cpu = torch.randint(0, 256, (n, group_size), device=device_cpu, dtype=torch.int32).contiguous().pin_memory()

        # Generate one random vector on GPU once, reuse for all p
        g_gpu = torch.Generator(device=device_gpu).manual_seed(seed)
        if mode == "mask":
            u = torch.rand(n, device=device_gpu, generator=g_gpu)
        else: # rearrange
            u = torch.randperm(n, device=device_gpu, generator=g_gpu)

        for p in mask_probs:
            if mode == "mask":
                mask = (u < p)
                n_selected = torch.count_nonzero(mask).item()
            else: # rearrange
                n_selected = int(n * p)
                indices = u[:n_selected]

            cpu_times = []
            gpu_times = []

            # Warmup
            for _ in range(2):
                if mode == "mask":
                    out = torch.empty((n_selected, group_size), dtype=x_cpu.dtype, device='cuda')
                    cuda_zero_copy_reader.zero_copy_mask(x_cpu, mask, out)
                else: # rearrange
                    out = torch.empty((n_selected, group_size), dtype=x_cpu.dtype, device='cuda')
                    cuda_zero_copy_reader.zero_copy_rearrange(x_cpu, indices, out)
                
                xg = x_cpu.to(device_gpu)
                _ = xg[indices] if mode == "rearrange" else xg[mask]
                torch.cuda.synchronize()

            # Timed loops for zero-copy path
            for _ in range(iters):
                if mode == "mask":
                    res1 = torch.empty((n_selected, group_size), dtype=x_cpu.dtype, device='cuda')
                    t0 = time.perf_counter_ns()
                    cuda_zero_copy_reader.zero_copy_mask(x_cpu, mask, res1)
                    # res1 = out_cpu[mask]
                else: # rearrange
                    res1 = torch.empty((n_selected, group_size), dtype=x_cpu.dtype, device='cuda')
                    t0 = time.perf_counter_ns()
                    cuda_zero_copy_reader.zero_copy_rearrange(x_cpu, indices, res1)
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                cpu_times.append(t1 - t0)

            # Timed loops for H2D copy path
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                xg = x_cpu.to(device_gpu)
                res2 = xg[indices] if mode == "rearrange" else xg[mask]
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                gpu_times.append(t1 - t0)

            if not torch.equal(res1, res2):
                 raise RuntimeError(f"Sanity check failed for gs={group_size}, p={p}: zero-copy path != copy+op")

            med_cpu_ms = stats.median(cpu_times) / 1e6
            med_gpu_ms = stats.median(gpu_times) / 1e6
            print(f"p={p:<7.4f} gs={group_size:<4} median_cpu={med_cpu_ms:.3f} ms   median_gpu={med_gpu_ms:.3f} ms")
            results.append((int(group_size), float(p), float(med_cpu_ms), float(med_gpu_ms)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Plot median latency vs p for zero-copy vs copy+op")
    parser.add_argument("--mode", type=str, default="rearrange", choices=["rearrange", "mask"],
                        help="Benchmark mode: 'rearrange' or 'mask'")
    parser.add_argument("--bytes", type=int, required=True,
                        help="Total tensor size in bytes (int32, 4B/elem)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Iterations per (p, gs) point (default 20)")
    parser.add_argument("--seed", type=int, default=123456,
                        help="RNG seed (default 123456)")
    parser.add_argument("--ps", type=float, nargs="+",
                        default=[1, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
                        help="Mask probabilities to test")
    parser.add_argument("--gs", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16],
                        help="Group sizes to test")
    parser.add_argument("--out-prefix", type=str, default="median_vs_p",
                        help="Prefix for CSV/PNG outputs")
    args = parser.parse_args()

    ps = sorted(list(args.ps), reverse=True)
    gs = sorted(list(args.gs))
    
    results = run_benchmark(args.mode, args.bytes, ps, gs, args.iters, seed=args.seed)

    # Save CSV
    csv_path = f"{args.out_prefix}_{args.mode}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("group_size,p,median_cpu_ms,median_gpu_ms\n")
        for r_gs, r_p, m_cpu, m_gpu in results:
            f.write(f"{r_gs},{r_p},{m_cpu},{m_gpu}\n")
    print(f"\nSaved CSV -> {csv_path}")

    # Plot
    results_by_gs = defaultdict(list)
    for r_gs, r_p, m_cpu, m_gpu in results:
        results_by_gs[r_gs].append((r_p, m_cpu, m_gpu))

    for group_size, data in results_by_gs.items():
        # Sort by p descending for a cleaner line
        data_sorted = sorted(data, key=lambda r: r[0], reverse=True)
        ps_sorted = [r[0] for r in data_sorted]
        cpu_sorted = [r[1] for r in data_sorted]
        gpu_sorted = [r[2] for r in data_sorted]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ps_sorted, cpu_sorted, marker="o", label="Zero-Copy Path (median)")
        ax.plot(ps_sorted, gpu_sorted, marker="o", label=f"H2D Copy + {args.mode} (median)")
        ax.set_xscale("log")
        ax.set_xlabel("Probability p")
        ax.set_ylabel("Median Latency (ms)")
        ax.set_title(f"Latency vs. Selectivity (mode={args.mode}, gs={group_size})\n"
                     f"Total Bytes: {args.bytes / (1024**3):.2f} GB, iters={args.iters}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        
        png_path = f"{args.out_prefix}_{args.mode}_gs{group_size}.png"
        fig.savefig(png_path, dpi=160)
        print(f"Saved plot -> {png_path}")


if __name__ == "__main__":
    main()