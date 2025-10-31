from sklearn.datasets import make_friedman1
import torch
import time
import pandas as pd

data = pd.DataFrame(columns=["width", "old", "new"])

torch.manual_seed(42)

N = 100_000_000
K = 4

for num in range(1, 8):
  print ("\nnum = ", num)
  keys = torch.randint(0, K, (N, num), dtype=torch.int64, device='cuda')
  # keys = torch.randint(0, K, (N,), dtype=torch.int64, device='cuda')
  # print (keys)
  values = torch.randn(N, dtype=torch.float64, device='cuda')
  torch.cuda.synchronize()

  round = 3
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
    old = end - start
    print(f"Aggregation time: {old:.6f} s")

    print ("[old] Key = ", unique_keys)
    print ("[old] Agg = ", agg)
    old_agg = agg
    # p = (mid - start) / (end - start) * 100
    # print(f"Proportion: {p:.2f}% items/s")

    print ("\n\n")
    start = time.time()
    mod = 9982444853
    print ("Key = ", keys)
    hash_coeff = [torch.randint(0, 10000000, (), dtype=torch.int64, device='cuda') for i in range(num)]    
    hash_tensor = 0
    for k in range(num):
      hash_tensor += keys[:, k] * hash_coeff[k]
      hash_tensor = hash_tensor % mod
    print ("Hash Tensor = ", hash_tensor)
    torch.cuda.synchronize()
    mid1 = time.time()
    print(f"Hashing time: {mid1 - start:.6f} s")

    unique_hash, inverse = torch.unique(hash_tensor, dim=0, return_inverse=True)
    torch.cuda.synchronize()
    
    # mid2 = time.time()
    # print(f"Unique (new) time: {mid2 - mid1:.6f} s")

    agg = torch.zeros(unique_keys.size(0), dtype=values.dtype, device='cuda')
    agg.scatter_add_(0, inverse, values)

    unique_key = torch.zeros((unique_hash.size(0), num), dtype=keys.dtype, device='cuda')
    unique_key.scatter_(0, inverse.unsqueeze(1).expand(-1, num), keys)

    torch.cuda.synchronize()

    end = time.time()
    new = end - start
    print(f"New Aggregation time: {new:.6f} s")

    print ("[new] Key = ", unique_key)
    print ("[new] Agg = ", agg)

    if _ + 1 == round:
      data = pd.concat([data, pd.DataFrame({"width": [num], "old": [old], "new": [new]})], ignore_index=True)

    old_agg = torch.sort(old_agg)[0]
    agg = torch.sort(agg)[0]
    assert torch.allclose(old_agg, agg)
    # if _ + 1 == round:
    #   # data = pd.concat([data, pd.DataFrame({"width": [num], "percent": [p]})], ignore_index=True)
    #   runtime = pd.concat([runtime, pd.DataFrame({"width": [num], "time": [end - start]})], ignore_index=True)



data.to_csv("optimized_time.csv", index=False)


