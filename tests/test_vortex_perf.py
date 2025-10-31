from IO.vortex_pipeline import VortexPipeline
import IO.vortex as vortex
from IO.vortex import NaiveExchange
from IO.pinned_mem import PinnedMemory
from torch.profiler import profile, record_function, ProfilerActivity
import constants
import torch
import time
from utility.tensor_utils import resize_tensor_list, show_tensor_usage, clean_tensors
from utility.logger import perf_logger, message_logger 

import time
from contextlib import contextmanager
granularity = 20_000_000

@contextmanager
def timer(label: str = ""):
    """
    Usage:
        with timer("my block"):
            ...  # code to time
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{label}] elapsed {end - start:.4f}s")

def test_vortex_interface(src_h, src_d, dev, pinned_mem):
    dst_d = [torch.empty_like(src_h[0], device=dev)]
    dst_h = [torch.empty_like(src_d[0], device='cpu')]
    with perf_logger().time("Pinned_mem dst"): 
        dst_h = pinned_mem.allocate_list(dst_h)
    with perf_logger().time("Exchange Vortex time"):
        ex = vortex.exchange(granularity)
        ex.launch(dst_d,src_h, dst_h, src_d)
        ex.sync()

def test_cuda_interface(src_h, src_d, dev, pinned_mem):
    stream0 = torch.cuda.Stream(device=dev)
    dst_d = torch.empty_like(src_h[0], device=dev)
    dst_h = torch.empty_like(src_d[0], device='cpu').pin_memory()
    with perf_logger().time("Exchange Naive time"):
        with torch.cuda.stream(stream0):
            dst_d.copy_(src_h[0], non_blocking=True)
            dst_h.copy_(src_d[0], non_blocking=True)
            stream0.synchronize()


def test_naive_exchange(src, chunk_size_b):
    def id(*x):
        return x
    vp = VortexPipeline([src], [torch.int64], id, chunk_size=chunk_size_b)
    vp.do_exchange(granularity, enable_naive=True)
    return vp.get_result()

def test_vortex_exchange(src, chunk_size_b):
    def id(*x):
        return x
    vp = VortexPipeline([src], [torch.int64], id, chunk_size=chunk_size_b)
    vp.do_exchange(granularity, enable_naive=False)
    return vp.get_result()

def main():
    print(torch.get_num_threads(), torch.get_num_interop_threads())
    torch.set_num_threads(64)
    torch.set_num_interop_threads(64)
    message_logger().info("Hi")
    
    # for i in range(10):
    #     with perf_logger().time("preprocess"):
    #         pinned_mem = PinnedMemory(60_000_000_000)
    #     dev = torch.device(f"cuda:0")
    #     src_h = [torch.randint(1, 100, (20_000_000_000 // 8,), dtype=torch.int64)]
    #     with perf_logger().time("src allocate"):
    #         # with profile(
    #         # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         # record_shapes=True,
    #         # profile_memory=True,
    #         # ) as prof:
    #         src_h = pinned_mem.allocate_list(src_h)
    #         pinned_mem.sync()
    #     src_d = [torch.randint(1, 100, (20_000_000_000 // 8,), dtype=torch.int64, device=dev)]
        
    #     print("vortex")
    #     test_vortex_interface(src_h, src_d, dev, pinned_mem)
    # #     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    #     print("cuda")
    #     test_cuda_interface(src_h, src_d, dev, pinned_mem)
    #     perf_logger().report()
    for _ in range(10):
        for array_size in range(9, 10):
            array_size_b = 10**array_size * 10
            src = torch.randint(1, 100, (array_size_b // 8,), dtype=constants.int_dtype)
            for chunk_size in range(9, 10):
                chunk_size_b = 10**chunk_size
                # show_tensor_usage()
                # start = time.time()
                test_vortex_exchange(src, chunk_size_b)
                # end = time.time()
                # del b
                # t = end - start
                # clean_tensors()
                print(f"Vortex, Chunk size={chunk_size_b}, array size = {array_size_b}, time = {0}")

                # start = time.time()
                test_naive_exchange(src, chunk_size_b)
                # end = time.time()
                # t = end - start
                # del a
                # clean_tensors()
                print(f"Naive, Chunk size={chunk_size_b}, array size = {array_size_b}, time = {0}")
    # for _ in range(10):
    #     src = torch.randint(1, 100, (1_000_000_000 // 8, ), dtype=torch.float64)
    #     dst = torch.empty_like(src)
    #     with timer("cpy"):
    #         dst.copy_(src)