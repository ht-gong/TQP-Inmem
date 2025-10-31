import sys
import os
import time
import torch
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from operators.filter import tqp_filter
from utility.logger import perf_logger, set_message_logger, set_perf_logger
from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np
import torch.profiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

from IO.pinned_mem import PinnedMemory

def test_filter():
    # size_l, size_r

    print("Pinning memory...", end="")
    pinned_pool = PinnedMemory(capacity_gb=200, block_size_mb= int(150 / 2000 * 1000))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(2))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 1024)
    print("done!")

    # set_exchange_to_naive()
    
    # for selectivity in [0.8, 0.8]:
    for selectivity in [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0]:
    # for selectivity in [0.4, 0.4]:

        rows = 300_000_0000
        set_perf_logger("./log.txt", True)
        set_message_logger("./log.txt", True)
        tensorsa = {}
        tensorsb = {}
        tensorsa = {
            1: Variable(torch.arange(rows), ''),
            101: Variable(torch.rand(rows, dtype=torch.float64), '')
        }

        for col_id in tensorsa.keys():
            orig_data = tensorsa[col_id].tensor
            _, tensorsa[col_id].tensor = pinned_pool.malloc_like(tensorsa[col_id].tensor)
            tensorsa[col_id].tensor[:] = orig_data
        
        threshold = 1 - selectivity
        args = {
            'input ids': [1, 101],
            'filter tree': ['AAA#101', '>', [f'{threshold}']]
        }
        print ("Start!")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.time()
        
        # if it != 2:
        res_rows = tqp_filter(True, tensorsb, tensorsa, args, None, mem_pool=pinned_pool, name="filter")
        torch.cuda.synchronize()
        print ("res_rows= ", res_rows)
        # else:
        #     with torch.profiler.profile(
        #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # CPU + GPU 事件
        #         schedule=schedule(wait=0, warmup=0, active=1),  # 采样调度
        #         on_trace_ready=tensorboard_trace_handler('./tests/joins'),    # 生成 TensorBoard 文件
        #         record_shapes=True,      # 记录 tensor 形状
        #         profile_memory=True,     # 记录内存使用
        #         with_stack=True          # 记录调用栈
        #     ) as profiler:
        #         res_rows = join_vortex(True, tensorsb, tensorsa, 2, 1, 1 << 26, [], 'inner')
        #         profiler.step()
        
        end = time.time()


        print (f"filter time = {end - start:.4f} s, {res_rows} rows")

        benchmark_result = {
            "time": end - start
        }
        
        with open(f"./tests/results/tqp_filter_{rows}rows_{int(selectivity*100)}select.json", 'w') as f:
        # with open(f"./tests/results/tqp_novortex_filter_{rows}rows_{int(selectivity*100)}select.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        perf_logger().report(f"./tests/results/tqp_log_filter_{rows}rows_{int(selectivity*100)}select.json")

        pinned_pool.free_all()

if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    test_filter()
