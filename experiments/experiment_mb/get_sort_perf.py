import torch, time
from IO.vortex import set_exchange_to_naive
from IO.vortex_pipeline import VortexPipeline
from utility.logger import perf_logger, set_perf_logger, set_message_logger
from IO.pinned_mem import PinnedMemory
import matplotlib.pyplot as plt
import gc
import pandas as pd
from hipTQP import merge
from utility.tensor_utils import find_boundaries_for_merge

datatype = torch.float32
GB = 1_000_000_000
mem_pool = PinnedMemory(250, 1_000)
chunk_size=1_000_000_000

def mask_op(t):
    mask = t > 0.5
    return t[mask], 0

def on_gpu_test(to_sort_device, name):
    to_sort_device, inv_index = torch.sort(to_sort_device)
    return to_sort_device, inv_index

def vortex_test(to_sort_host, name):
    def sort_op(*_key):
        srted, inv_index = torch.sort(_key[0], stable=True)
        return [srted, inv_index]

    def merge_op(*_to_split):
        _val = _to_split[:len(_to_split) // 2]
        _key = _to_split[len(_to_split) // 2:]
        _prevval = torch.empty(0, dtype=_val[0].dtype, device="cuda")
        prevind = torch.empty(0, dtype=_key[0].dtype, device="cuda")
        for v, k in zip(_val, _key):
          _nextval = torch.empty(v.numel() + _prevval.numel(), dtype=_val[0].dtype, device="cuda")
          nextind = torch.empty(k.numel() + prevind.numel(), dtype=_key[0].dtype, device="cuda")
        #   print(_prevval, v, _nextval, prevind, k, nextind)
          merge(_prevval, v, _nextval, prevind, k, nextind)
          _prevval, prevind = _nextval, nextind
        return [nextind]
    sort_pipeline = VortexPipeline([to_sort_host], [torch.int64, torch.int64], sort_op, mem_pool, chunk_size=chunk_size, name=f'{name} sort')
    sort_pipeline.do_exchange(20_000_000)
    res = sort_pipeline.get_result()
    srted = [t[0] for t in res]
    inv_index = [t[1] for t in res]
    chunk_bounds = find_boundaries_for_merge(srted, chunk_size)
    merge_pipeline = VortexPipeline(srted + inv_index, [torch.int64], merge_op, mem_pool,  div=chunk_bounds * 2, chunk_size=chunk_size, name=f'{name} merge')
    merge_pipeline.do_exchange(20_000_000)
    res = merge_pipeline.get_result()
    return res

def no_vortex_test(to_sort_host, name):
    set_exchange_to_naive()
    return vortex_test(to_sort_host, name)

def on_cpu_test(to_sort_host, name):
    to_sort_host, inv_index = torch.sort(to_sort_host)
    return to_sort_host, inv_index

methods = {
    "GPU": on_gpu_test,
    "CPU": on_cpu_test,
    "Vortex": vortex_test,
    "No Vortex": no_vortex_test
}



def main():

    # sizes_gb = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 20, 30, 50, 70, 100]
    sizes_gb = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 20, 30]
    # sizes_gb = [50]
    results = {name: [] for name in methods}
    results["Vortex transfer"] = []
    results["Vortex compute"] = []
    results["No Vortex transfer"] = []
    results["No Vortex compute"] = []

    set_message_logger(enable=True)
    set_perf_logger(enable=True)


    numel = int(0.1 * GB) // 4  # float32 = 4 bytes
    _, to_sort = mem_pool.malloc(numel, datatype)
    to_sort[:] = torch.load(f'experiments/experiment_mb/tensor_{0.1}GB.pth', map_location='cpu')
    _ = vortex_test(to_sort, "test")
    _ = on_cpu_test(to_sort, "test")

    for name, fn in methods.items():
        for gb in sizes_gb:
            # only run on_gpu_test for small sizes
            if name == "GPU" and gb >= 10:
                break

            if name == "CPU" and gb >= 30:
                break

            # prepare data
            numel = int(gb * GB) // 4  # float32 = 4 bytes
            _, to_sort = mem_pool.malloc(numel, datatype)
            to_sort[:] = torch.load(f'experiments/experiment_mb/tensor_{gb}GB.pth', map_location='cpu')
            if name == "GPU":
                to_sort = to_sort.to('cuda')

            pipeline_name = f"{name}_{gb}"
            # time it
            # deadlock()
            start = time.perf_counter()
            res = fn(to_sort, pipeline_name)
            torch.cuda.synchronize()
            stop = time.perf_counter()
            elapsed = stop - start

            gc.collect()
            torch.cuda.empty_cache()

            results[name].append(elapsed)

            if name== "Vortex" or name == "No Vortex":
                results[f"{name} transfer"].append(perf_logger()[f"{pipeline_name} transfer"]['total'])
                results[f"{name} compute"].append(perf_logger()[f"{pipeline_name} apply op"]['total'])

            mem_pool.free_all()
            print(f"{name}: {gb} GB → {elapsed:.4f} s")


    # build one Series per method, indexed by the subset of sizes it actually ran on
    series_dict = {
        name: pd.Series(times, index=sizes_gb[:len(times)])
        for name, times in results.items()
    }

    # assemble into DataFrame — missing will be NaN
    df = pd.DataFrame(series_dict)
    df.index.name = 'Size_GB'
    df.columns.name = 'Method'

    print(df)

    out_path = "experiments/experiment_mb/sorting_results.csv"
    df.to_csv(out_path)
    print(f"Results saved to {out_path}")
    # # --- now plot ---
    # for name, times in results.items():
    #     # sizes might be shorter for GPU sort
    #     xs = sizes_gb[:len(times)]
    #     plt.plot(xs, times, label=name)

    # plt.xlabel("Data Size (GB)")
    # plt.ylabel("Time (s)")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title("Masking Performance Comparison")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("performance_comparison_mask.png", format="png", dpi=300, bbox_inches="tight")
