import sys
import os
import time
import torch
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from operators.join import tqp_join_new
from IO.vortex import set_exchange_to_naive
from utility.logger import perf_logger, set_message_logger, set_perf_logger
from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np
import torch.profiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

from IO.pinned_mem import PinnedMemory

def test_inner():
    # size_l, size_r
    print("Pinning memory...", end="")
    pinned_pool = PinnedMemory(capacity_gb=75, block_size_mb= int(150 / 2000 * 1000))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(2))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 1024)
    print("done!")

    # set_exchange_to_naive()

    # for R in [1, 1]:
    for rows in [10**9, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:
        R = 1
    # for it, R in enumerate([1, 1, 1]):
        torch.cuda.empty_cache()
        # rows = 1_0000_00000
        # rows = 5_0000_0000
        set_perf_logger("./log.txt", True)
        set_message_logger("./log.txt", True)

        tensorsa = {
            1: Variable((torch.arange(rows, dtype=torch.int32) % (rows // R)), ''),
            101: Variable(torch.randn(rows, dtype=torch.float64), '')
        }
        tensorsb = {
            2: Variable(torch.arange(rows, dtype=torch.int32), ''),
            102: Variable(torch.randn(rows, dtype=torch.float64), '')
        }

        for col_id in tensorsa.keys():
            orig_data = tensorsa[col_id].tensor
            _, tensorsa[col_id].tensor = pinned_pool.malloc_like(tensorsa[col_id].tensor)
            tensorsa[col_id].tensor[:] = orig_data
        
        for col_id in tensorsb.keys():
            orig_data = tensorsb[col_id].tensor
            _, tensorsb[col_id].tensor = pinned_pool.malloc_like(tensorsb[col_id].tensor)
            tensorsb[col_id].tensor[:] = orig_data

        print ("Start!")

        start = time.time()
        
        res_rows = tqp_join_new(True, tensorsb, tensorsa, 2, 1, mem_pool=pinned_pool)  

        torch.cuda.synchronize()
        end = time.time()


        print (f"join time = {end - start:.4f} s, {res_rows} rows")

        benchmark_result = {
            "time": end - start
        }
        # name = f"tqp_newjoin_naive_{R}repeats_{rows}rows"
        name = f"tqp_newjoin_vortex_{R}repeats_{rows}rows"
        with open(f"./tests/results/{name}.json", 'w') as f:
        # with open(f"./tests/results/tqp_{R}repeats_{rows}rows.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        perf_logger().report(f"./tests/results/log_{name}.json")
        pinned_pool.free_all()


if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    test_inner()
