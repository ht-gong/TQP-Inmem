import torch, time
from IO.vortex import set_exchange_to_naive
from IO.vortex_pipeline import VortexPipeline, out_of_core_pipeline
from utility.logger import perf_logger, set_perf_logger, set_message_logger
from IO.pinned_mem import PinnedMemory
from typing import List, Literal
import matplotlib.pyplot as plt
import gc
import pandas as pd
from conversion import append_nulls, date_to_float, get_id, get_literals, is_date, is_float, str_to_np, rearrange_tensors, normalize
from IO import vortex_pipeline
from operators.filter import evaluate
from variable import Variable

datatype = torch.float32
GB = 1_000_000_000
mem_pool = PinnedMemory(250, 50)

def evaluate_join(device_name, tensorsa, tensorsb, tree, leftidx, rightidx):
  def leaf(id: int):
      if id in tensorsa.keys():
        return tensorsa[id].tensor[leftidx].squeeze()
      else:
        return tensorsb[id].tensor[rightidx].squeeze()
      
  return evaluate(device_name, None, tree, leftidx.size(0), None, leaf)

def join_kernel(tensorsa, tensorsb, leftcol, rightcol, hashtable_size, condition_list, type: Literal['inner', 'right-semi', 'right-anti', 'right-outer']):
  assert type in {'inner', 'right-semi', 'right-anti', 'right-outer'}, "Invalid Join type."

  m = hashtable_size
  
  left = normalize(tensorsa[leftcol].tensor)
  right = normalize(tensorsb[rightcol].tensor)

  leftIdx = torch.arange(left.size(0), device='cuda:0')
  rightIdx = torch.arange(right.size(0), device='cuda:0')

  leftHash = torch.remainder(left, m)
  rightHash = torch.remainder(right, m)

  hashBincount = torch.bincount(leftHash) if leftHash.size(0) > 0 else torch.zeros(1, dtype=torch.int64, device=device_name)
  maxHashBucket = torch.max(hashBincount)

  leftHash = leftHash.to(torch.int64)

  leftOutList = []
  rightOutList = []

  if type == 'right-outer':
      matched = torch.zeros(right.size(0), device='cuda:0')
  
  for i in range(maxHashBucket):
      init_v = m + 1
      if type in {'right-semi', 'right-anti'}:
          init_v += 1
      hashTable = torch.full((init_v,), -1, device='cuda:0')
      hashTable.scatter_(0, leftHash, leftIdx)

      leftIdxSct = torch.masked_select(hashTable, hashTable >= 0)
      leftHash[leftIdxSct] = m 
      
      leftCandIdx = hashTable[rightHash]
      validKeyMask = leftCandIdx >= 0
      validLeftIdx = torch.masked_select(leftCandIdx, validKeyMask)
      validRightIdx = torch.masked_select(rightIdx, validKeyMask)

      matchMask = left[validLeftIdx] == right[validRightIdx]
      for condition in condition_list:
          matchMask.logical_and_(evaluate_join('cuda:0', tensorsa, tensorsb, condition, validLeftIdx, validRightIdx))

      rightMatchIdx = torch.masked_select(validRightIdx, matchMask)

      if type == 'right-outer':
          matched[rightMatchIdx] = True
      elif type in {'right-semi', 'right-anti'}:
          rightHash[rightMatchIdx] = m + 1
      
      if type != 'right-anti':
          rightOutList.append(rightMatchIdx)

      if type in {'inner', 'right-outer'}:
          leftMatchIdx = torch.masked_select(validLeftIdx, matchMask)
          leftOutList.append(leftMatchIdx)

  if type == 'right-outer':
      rightOutList.append(rightIdx[matched == False])

  if type == 'right-anti':
      rightOutIdx = rightIdx[rightHash < m]
  else:
      rightOutIdx = torch.cat(rightOutList) if rightOutList else torch.tensor([], dtype=torch.int64, device='cuda:0')
  
  
  if type in {'inner', 'right-outer'}:
    leftOutIdx = torch.cat(leftOutList) if leftOutList else torch.tensor([], dtype=torch.int64, device='cuda:0')
    return leftOutIdx.contiguous(), rightOutIdx.contiguous()

  if vortex_pipeline.global_gpu_tensor is None:
     vortex_pipeline.global_gpu_tensor = [rightOutIdx]
  else:
    vortex_pipeline.global_gpu_tensor[0] = torch.cat((vortex_pipeline.global_gpu_tensor[0], rightOutIdx), dim=0)
  return (torch.tensor([], dtype=torch.int64, device='cuda'),)

def mask_op(t):
    mask = t > 0.5
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=0)  
    print(f"GPU0 free: {free_bytes/1024**2:.1f} MiB / {total_bytes/1024**2:.1f} MiB")
    return t[mask]

def sort_op(t):
    return torch.sort(t)[0]

def join_op(t):
    res = join_kernel({0: Variable(t[:t.shape[0]//2], 'int')}, {1: Variable(t[t.shape[0]//2:], 'int')}, 0, 1, 1 << 27, [], 'inner')
    print(res)
    return res[0]

def agg_op(t):
    N = t.shape[0]
    count = torch.arange(5, device='cuda:0').repeat_interleave(N // 5)
    res = torch.bincount(count, weights=t)
    return res

compute_fns = [mask_op, join_op, agg_op, sort_op]
merge_fns = [None, None, None, None]

compute_fn = mask_op 
merge_fn = None

def on_gpu_test(to_process_device, name, chunk_size):
    res_device, _ = compute_fn(to_process_device)
    return res_device 

def vortex_test(to_process_host, name, chunk_size):
    def on_dev_compute(*args):
        args = list(args)
        return [compute_fn(args[0])]
    if chunk_size < 8_000_000_000:
        vp = VortexPipeline([to_process_host], [datatype], on_dev_compute, mem_pool, name=name, chunk_size=chunk_size)
        vp.do_exchange(20_000_000)
        res = vp.get_result()
    else:
        res = out_of_core_pipeline(to_process_host, on_dev_compute, mem_pool, chunk_size, 'cuda:0', 'cuda:1', name=name)
        print(res)
    return res

def no_vortex_test(to_process_host, name, chunk_size):
    set_exchange_to_naive()
    return vortex_test(to_process_host, name, chunk_size=chunk_size)

def on_cpu_test(to_process_host, name, chunk_size):
    to_process_host = compute_fn(to_process_host)
    return to_process_host

methods = {
    # "GPU": on_gpu_test
    # "CPU": on_cpu_test,
    # "Vortex": vortex_test,
    "No Vortex": no_vortex_test
}



def main():
    global datatype 
    # sizes_gb = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 20, 30, 50, 70, 100]
    sizes_gb = [50]
    chunk_sizes_mb = [150, 200, 300, 500, 1000, 2000, 3000, 5000, 8000]
    results = {name: [] for name in methods}
    set_message_logger(enable=True)
    set_perf_logger(enable=True)

    if compute_fn != join_op:
        numel = int(0.01 * GB) // 4  # float32 = 4 bytes
        _, to_sort = mem_pool.malloc(numel, datatype)
        to_sort[:] = torch.load(f'experiments/experiment_mb/tensor_{0.01}GB.pth', map_location='cpu')
        _ = vortex_test(to_sort, "test", 1_000_000_000)
    # _ = on_cpu_test(to_sort, "test", 1_000_000_000)

    for name, fn in methods.items():
        for gb in sizes_gb:
            # only run on_gpu_test for small sizes
            if name == "GPU" and gb >= 10:
                break

            if name == "CPU" and gb >= 10:
                break
            
            for chunk_size in chunk_sizes_mb:
                chunk_size_b = chunk_size * 1_000_000

                if compute_fn == join_op:
                    numel = int(25 * GB) // 4
                    datatype = torch.int64
                    _, to_sort = mem_pool.malloc(numel * 2, datatype)
                    to_sort[: numel] = torch.load(f'experiments/experiment_mb/tensor_{25}GB.pth', map_location='cpu')
                    to_sort[numel:] = to_sort[: numel].clone()
                    print(to_sort)
                # prepare data
                else:
                    numel = int(gb * GB) // 4  # float32 = 4 bytes
                    _, to_sort = mem_pool.malloc(numel, datatype)
                    to_sort[:] = torch.load(f'experiments/experiment_mb/tensor_{gb}GB.pth', map_location='cpu')
                    if name == "GPU":
                        to_sort = to_sort.to('cuda')

                pipeline_name = f"{name}_{gb}"
                # time it
                # deadlock()
                start = time.perf_counter()
                _ = fn(to_sort, pipeline_name, chunk_size_b)
                # torch.cuda.synchronize()
                stop = time.perf_counter()
                elapsed = stop - start

                gc.collect()
                torch.cuda.empty_cache()

                results[name].append(elapsed)

                mem_pool.free_all()
                print(f"{name}: {gb} GB → {elapsed:.4f} s")


    # build one Series per method, indexed by the subset of sizes it actually ran on
    series_dict = {
        name: pd.Series(times, index=chunk_sizes_mb[:len(times)])
        for name, times in results.items()
    }

    # assemble into DataFrame — missing will be NaN
    df = pd.DataFrame(series_dict)
    df.index.name = 'chunk_sizes_mb'
    df.columns.name = 'Method'

    print(df)

    out_path = "experiments/experiment_mb/chunking_results_join.csv"
    df.to_csv(out_path)
    # print(f"Results saved to {out_path}")
    # # --- now plot ---
    # for name, times in results.items():
    #     # sizes might be shorter for GPU sort
    #     xs = sizes_gb[:len(times)]
    #     plt.plot(xs, times, label=name)

    # plt.xlabel("Chunk Size (GB)")
    # plt.ylabel("Time (s)")
    # plt.title("Chunking Size and Performance Comparison Join")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("performance_comparison_join.png", format="png", dpi=300, bbox_inches="tight")
