import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import time
import torch
import pytest
import json


# from operators.filter import tqp_filter
# from utility.logger import perf_logger, set_message_logger, set_perf_logger
# from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np
# import torch.profiler
# from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

def test_filter():
    # size_l, size_r
    for selectivity in [0.2, 1.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        rows = 1_0000_00000
        # 
        # set_perf_logger("./log.txt", True)
        # set_message_logger("./log.txt", True)
        tensorsa = {}
        tensorsb = {}
        torch.cuda.empty_cache()
        tensorsa = {
            1: Variable((torch.arange(rows)).pin_memory(), ''),
            101: Variable(torch.rand(rows, dtype=torch.float64).pin_memory(), '')
        }
        
        threshold = 1 - selectivity
        args = {
            'input ids': [1, 101],
            'filter tree': ['AAA#101', '>', [f'{threshold}']]
        }
        print ("Start!")
        
        # print("Num threads:", torch.get_num_threads())
        # print("Num inter-op threads:", torch.get_num_interop_threads())
        start = time.time()
        
        tensorsa[1].tensor = tensorsa[1].tensor.to('cuda')
        tensorsa[101].tensor = tensorsa[101].tensor.to('cuda')
        torch.cuda.synchronize()

        time_1 = time.time()
        mask = torch.gt(tensorsa[101].tensor, threshold)
        
        tensorsa[1].tensor = tensorsa[1].tensor[mask]
        tensorsa[101].tensor = tensorsa[101].tensor[mask]

        torch.cuda.synchronize()
        time_2 = time.time()
        print ("kernel = ", time_2 - time_1)
        
        tensorsa[1].tensor = tensorsa[1].tensor.to('cpu')
        tensorsa[101].tensor = tensorsa[101].tensor.to('cpu')

        torch.cuda.synchronize()

        res_rows = tensorsa[1].tensor.size(0)
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


        print ("kernel time = ", time_2 - time_1)
        print (f"filter time = {end - start:.4f} s, {res_rows} rows")
        benchmark_result = {
            "time": end - start
        }
        benchmark_result_kernel = {
            "time": time_2 - time_1
        }
        
        # with open(f"./tests/results/hand_filter_cpu_{rows}rows_{int(selectivity*100)}select.json", 'w') as f:
        #     json.dump(benchmark_result, f, indent=2)
        
        with open(f"./tests/results/hand_filter_{rows}rows_{int(selectivity*100)}select.json", 'w') as f:
            json.dump(benchmark_result, f, indent=2)
        with open(f"./tests/results/hand_filter_gpu_{rows}rows_{int(selectivity*100)}select.json", 'w') as f:
            json.dump(benchmark_result_kernel, f, indent=2)
        # perf_logger().report(f"./tests/results/tqp_log_filter_{rows}rows_{int(selectivity*100)}select.json")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    
    test_filter()
