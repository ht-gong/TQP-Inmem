from sqlite3 import Time
import sys
import os
import time
import torch
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from IO import vortex
from operators.sort import ooc_sort_wrapper_simple
from IO.vortex import set_exchange_to_naive
from utility.logger import perf_logger, set_message_logger, set_perf_logger
from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np
# import torch.profiler
# from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

from IO.pinned_mem import PinnedMemory

def test_inner():
    # size_l, size_r
    print("Pinning memory...", end="")
    pinned_pool = PinnedMemory(capacity_gb=32*3, block_size_mb= int(32 / 2000 * 1000))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(2))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 1024)
    print("done!")

    # set_exchange_to_naive()
    # 1e9 rows, 
    for R in [1, 1, 1]:
        rows = 1_0000_0000_0
        set_perf_logger("./log.txt", True)
        set_message_logger("./log.txt", True)

        _, t = pinned_pool.malloc(rows, torch.int32)
        t[:] = torch.randint(0, rows * 2, (rows,), dtype=torch.int32)

        print ("Start!")
        # set_exchange_to_naive()
        exchange = vortex.exchange(20_000_000)

        start = time.time()
        
        # ooc_sort_wrapper_simple(t, chunk_size=1_000_000_000, mem_pool=pinned_pool)
        print (t.is_pinned(), " pin?")
        t_gpu = torch.empty_like(t, device='cuda')
        exchange.launch([t_gpu], [t], [torch.tensor([])], [torch.tensor([], device='cuda')])
        exchange.sync()

        sorted_tensor, inv_index = torch.sort(t_gpu, stable=True, descending=False)
        inv_index = inv_index.to(torch.int32)
        torch.cuda.synchronize()
        _, sorted_tensor_cpu = pinned_pool.malloc_like(sorted_tensor)
        _, inv_index_cpu = pinned_pool.malloc_like(inv_index)

        exchange.launch([torch.tensor([], device='cuda')], [torch.tensor([])], [sorted_tensor_cpu, inv_index_cpu], [sorted_tensor, inv_index])
        exchange.sync()
        

        end = time.time()


        print (f"join time = {end - start:.4f} s, rows")

        benchmark_result = {
            "time": end - start
        }
        
        # with open(f"./tests/results/tqp_vortex_{R}repeats_{rows}rows.json", 'w') as f:
        # with open(f"./tests/results/tqp_{R}repeats_{rows}rows.json", 'w') as f:
            # json.dump(benchmark_result, f, indent=2)
        # perf_logger().report(f"./tests/results/tqp_log_{R}repeats_{rows}rows.json")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    test_inner()
