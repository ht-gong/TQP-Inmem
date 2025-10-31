# benchmark_naive_exchange.py
import torch, time, math, argparse, json, pathlib, matplotlib.pyplot as plt
from utility.logger import set_datasize_logger
from IO.vortex import NaiveExchange                    # import your class

TOTAL_BYTES = 1 << 27        # 128 MB
DTYPE        = torch.float64  # 8-byte element → easy size arithmetic

def make_tensors(chunk_bytes: int):
    """Split 1 GiB into N chunks of size `chunk_bytes` (rounded down)."""
    n_chunks   = TOTAL_BYTES // chunk_bytes
    elems      = chunk_bytes // torch.tensor([], dtype=DTYPE).element_size()

    # Host tensors – use pin_memory=True for fastest H↔D traffic
    src_host   = [torch.empty(elems, dtype=DTYPE, pin_memory=True) for _ in range(n_chunks)]
    dst_host   = [torch.empty(elems, dtype=DTYPE, pin_memory=True) for _ in range(n_chunks)]

    # Device tensors
    src_dev    = [torch.empty(elems, dtype=DTYPE, device="cuda") for _ in range(n_chunks)]
    dst_dev    = [torch.empty(elems, dtype=DTYPE, device="cuda") for _ in range(n_chunks)]

    return src_host, dst_host, src_dev, dst_dev

def run_one(granularity):
    xfer = NaiveExchange(20_000_000)
    src_host, dst_host, src_dev, dst_dev = make_tensors(granularity)

    print(f"running {granularity}")
    torch.cuda.synchronize()            # ensure clean start
    t0 = time.perf_counter()
    xfer.launch(dst_dev, src_host, dst_host, src_dev)
    xfer.sync()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    bw = TOTAL_BYTES * 2 / dt / (1<<30)     # GiB per second
    
    print(f"finished {granularity}")
    del src_host, dst_host, src_dev, dst_dev
    return {"granularity": granularity, "seconds": dt, "bandwidth_gib_s": bw}

def main():
    set_datasize_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["64", "512", "4K", "64K", "128K", "256K", "512K", "1M", "4M", "16M", "128M"],
                        help="chunk sizes (e.g. 8 4K 1M 4M). K/M/G imply powers of two.")
    parser.add_argument("--plot", action="store_true", help="draw matplotlib graph")
    args = parser.parse_args()


    fer = NaiveExchange(TOTAL_BYTES)
    src_host, dst_host, src_dev, dst_dev = make_tensors(TOTAL_BYTES)
    print("tensor allocated")
    # Warm-up once to get CUDA context + pageable→pinned allocs out of the way
    fer.launch(dst_dev, src_host, dst_host, src_dev)
    fer.sync()

    print("transfer done")

    def parse_size(s):
        multipliers = {"K": 1<<10, "M": 1<<20, "G": 1<<30}
        if s[-1].upper() in multipliers:
            return int(s[:-1]) * multipliers[s[-1].upper()]
        return int(s)

    sizes = [parse_size(s) for s in args.sizes]
    results = [run_one(sz) for sz in sizes]

    # Pretty print & save json
    for r in results:
        g = r["granularity"]
        print(f"{g:>8,d} bytes  →  {r['seconds']:.4f} s   {r['bandwidth_gib_s']:.2f} GiB/s")
    # pathlib.Path("granularity_results.json").write_text(json.dumps(results, indent=2))

    if args.plot:
        xs = [r["granularity"] / (1<<20) for r in results]
        ys = [r["bandwidth_gib_s"] for r in results]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xscale("log")
        plt.xlabel("Chunk size (MB, log scale)")
        plt.ylabel("Effective bandwidth (GiB/s)")
        plt.title("PCIE host <-> device transfer speed vs. granularity")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        path = "experiments/figures/granularity_bandwidth.png"
        plt.savefig(path, dpi=200)
        print("plot saved to ", path)
        plt.show()

if __name__ == "__main__":
    main()
