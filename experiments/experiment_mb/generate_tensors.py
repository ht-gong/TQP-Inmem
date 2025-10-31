#!/usr/bin/env python3
import torch
import sys

def generate_and_save_tensors_gb(sizes_gb, dtype=torch.float32, device='cpu', out_prefix='tensor'):
    # bytes per element (float32 → 4 bytes)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    # define 1 GB = 1e9 bytes
    BYTES_PER_GB = 1_000_000_000

    for gb in sizes_gb:
        numel = int(gb * BYTES_PER_GB) // bytes_per_elem
        print(f"[{gb} GB] → {numel:,} elements", file=sys.stderr)

        try:
            # allocate random data to force actual memory usage
            # t = torch.randn(numel, dtype=dtype, device=device)
            t = torch.arange(numel, dtype=torch.int64, device='cpu')
        except RuntimeError as e:
            print(f"  ⚠️  Failed to allocate {gb} GB: {e}", file=sys.stderr)
            continue

        fn = f"experiments/experiment_mb/{out_prefix}_{gb}GB.pth"
        torch.save(t, fn)
        print(f"  ✔️  Saved {fn}", file=sys.stderr)

def main():
    sizes = [25]  

    # you could also parse from command-line:
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('sizes', nargs='+', type=int, help="Sizes in GB")
    # args = parser.parse_args()
    # sizes = args.sizes

    generate_and_save_tensors_gb(sizes)


if __name__ == "__main__":
    # example sizes; change or extend as you like
    main()

