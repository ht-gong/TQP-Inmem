import torch
import time
import pandas as pd

data = pd.DataFrame(columns=["width", "percent"])
runtime = pd.DataFrame(columns=["width", "time"])

torch.manual_seed(42)

N = 10_000_000
# N = 10
K = 4

for num in range(1, 12):

  keys = torch.randint(0, K, (N, num), dtype=torch.int64, device='cuda')
  # keys = torch.randint(0, K, (N,), dtype=torch.int64, device='cuda')
  # print (keys)
  values = torch.randn(N, dtype=torch.float64, device='cuda')
  torch.cuda.synchronize()

  round = 5
  for _ in range(round):
    start = time.time()

    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    # print (unique_keys, inverse)
    torch.cuda.synchronize()
    mid = time.time()

    print (f"Unique time: {mid - start:.6f} s")
    agg = torch.zeros(unique_keys.size(0), dtype=values.dtype, device='cuda')
    agg.scatter_add_(0, inverse, values)
    torch.cuda.synchronize()

    end = time.time()
    print(f"Aggregation time: {end - start:.6f} s")
    p = (mid - start) / (end - start) * 100
    print(f"Proportion: {p:.2f}% items/s")

    if _ + 1 == round:
      data = pd.concat([data, pd.DataFrame({"width": [num], "percent": [p]})], ignore_index=True)
      runtime = pd.concat([runtime, pd.DataFrame({"width": [num], "time": [end - start]})], ignore_index=True)

data.to_csv("experiment_results.csv", index=False)
runtime.to_csv("experiment_runtime.csv", index=False)

  # print(f"Number of unique keys: {unique_keys.numel()}")
