import sys
import os
import time
import torch
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from IO.vortex import set_exchange_to_naive
from IO.pinned_mem import PinnedMemory
from utility.logger import get_mem_consumption, perf_logger, set_message_logger, set_perf_logger
from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np

# 1e9, h = 30, 8.26 s,
def test_inner():
    # size_l, size_r
    print("Pinning memory...", end="")
    pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(200 / 2000 * 1000))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 5)
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(2))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 1024)
    print("done!")
    # set_exchange_to_naive()
    # for rows in [10**8, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:
    for rows in [10**9]:
        for R in [1]:
            set_perf_logger("./log.txt", True)
            set_message_logger("./log.txt", True)

            tensorsa = {
                1: Variable((torch.arange(rows) % (rows // R)), ''),
                101: Variable(torch.randn(rows), '')
            }
            tensorsb = {
                2: Variable(torch.arange(rows), ''),
                102: Variable(torch.randn(rows), '')
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
            h = 30
            res_rows = join_vortex(True, tensorsb, tensorsa, 2, 1, 1 << 30, [], 'inner', mem_pool=pinned_pool)
            torch.cuda.synchronize()
            end = time.time()


            print (f"join time = {end - start:.4f} s, {res_rows} rows")

            benchmark_result = {
                "time": end - start
            }

            print (get_mem_consumption())
            
            with open(f"./tests/results/tqp_log_{R}repeats_{rows}rows_{h}htsize.json", 'w') as f:
            # with open(f"./tests/results/tqp_novortex_{R}repeats_{rows}rows_{26}htsize.json", 'w') as f:
                json.dump(benchmark_result, f, indent=2)

            perf_logger().report(f"./tests/results/log__tqp_{R}repeats_{rows}rows_{h}htsize.json")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    test_inner()
    