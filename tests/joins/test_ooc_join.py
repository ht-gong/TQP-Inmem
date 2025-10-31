import sys
import os
import time
import torch
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from IO.vortex import set_exchange_to_naive
from utility.logger import perf_logger, set_message_logger, set_perf_logger
from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np
import torch.profiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

def generate_zipfian_data(num_rows, num_keys, skew_factor=1.0001):
    zipfian_keys = np.random.zipf(skew_factor, num_rows) % num_keys
    return torch.tensor(zipfian_keys, dtype=torch.int64)
from IO.pinned_mem import PinnedMemory

def test_inner():
    # size_l, size_r
    print("Pinning memory...", end="")
    pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(150 / 2000 * 1000))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= int(2))
    # pinned_pool = PinnedMemory(capacity_gb=150, block_size_mb= 1024)
    print("done!")

    # set_exchange_to_naive()

    for R in [1, 1, 5, 10, 20]:
    # for it, R in enumerate([1, 1, 1]):
        rows = 1_0000_0000
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
        
        # if it != 2:
        res_rows = join_vortex(True, tensorsb, tensorsa, 2, 1, 1 << 26, [], 'inner', mem_pool=pinned_pool)  
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
        torch.cuda.synchronize()
        end = time.time()


        print (f"join time = {end - start:.4f} s, {res_rows} rows")

        benchmark_result = {
            "time": end - start
        }
        
        with open(f"./tests/results/tqp_vortex_{R}repeats_{rows}rows.json", 'w') as f:
        # with open(f"./tests/results/tqp_{R}repeats_{rows}rows.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        perf_logger().report(f"./tests/results/tqp_log_{R}repeats_{rows}rows.json")

def test_outer():
    tensorsb = {
        1: Variable(torch.tensor([1, 2, 5, 7, 9]), ''),
        11: Variable(torch.tensor([11, 22, 55, 77, 99]), '')
    }
    tensorsa = {
        2: Variable(torch.tensor([1, 3, 7, 8, 10]), ''),
        22: Variable(torch.tensor([6, 6, 6, 6, 6]), '')
    }
    print ("Start!")

    start = time.time()
    res_rows = join_vortex(True, tensorsa, tensorsb, 2, 1, 1 << 27, [], 'right-outer')

    end = time.time()
    print (f"join time = {end - start:.4f} s, {res_rows} rows")

    print (tensorsa[2].tensor[0:20])
    print (tensorsa[22].tensor[0:20])
    print (tensorsb[1].tensor[0:20])
    print (tensorsb[11].tensor[0:20])

def test_semi_anti():
    tensorsb = {
        1: Variable(torch.tensor([1, 2, 5, 7, 9]), ''),
        11: Variable(torch.tensor([11, 22, 55, 77, 99]), '')
    }
    tensorsa = {
        2: Variable(torch.tensor([1, 1, 7, 8, 10]), ''),
        22: Variable(torch.tensor([6, 6, 6, 6, 6]), '')
    }
    print ("Start!")

    start = time.time()
    res_rows = join_vortex(True, tensorsa, tensorsb, 2, 1, 1 << 27, [], 'right-anti') # 'right-semi'

    end = time.time()
    print (f"join time = {end - start:.4f} s, {res_rows} rows")

    print (tensorsb[1].tensor[0:20])
    print (tensorsb[11].tensor[0:20])

if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    test_inner()
