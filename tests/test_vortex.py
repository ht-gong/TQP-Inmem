import torch
import math
import json
import sys, os
import gc
import time
from IO.vortex_pipeline import VortexPipeline
import IO.vortex as vortex
from operators.sort import ooc_sort_wrapper
from utility.import_utils import load_hipTQP_lib
from utility.tensor_utils import tensor_lists_equal
hipTQP = load_hipTQP_lib()
import hipTQP

tensor_size_B = 800  # in bytes
chunk_size_B = 400  # in bytes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_vortex_sort(cols):
    cols = [c.to('cpu') for c in cols]
    
    torch.cuda.synchronize(device=device)

    start_time = time.time()
    perm = torch.arange(cols[0].numel(), device=device)
    for i, (col, desc) in enumerate(zip(reversed(cols), [True, True, False, True, False])): 
        # Sort each column on GPU
        perm = ooc_sort_wrapper(col, perm, desc, chunk_size_B)
        print(perm)

    for i in range(len(cols)):
        cols[i] = torch.gather(cols[i], 0, torch.arange(cols[0].numel(), device='cpu').flip(0))

    elapsed = time.time() - start_time
    torch.cuda.synchronize(device=device)
    return perm

def test_on_gpu_sort(cols): 
    gpu_cols = [c.to(device) for c in cols]
    
    torch.cuda.synchronize(device=device)

    start_time = time.time()
    perm = torch.arange(gpu_cols[0].numel(), device=device)
    for col, desc in zip(reversed(gpu_cols), [True, True, False, True, False]): 
        # Sort each column on GPU
        tmp = torch.gather(col, 0, perm)
        _, inverse_index = torch.sort(tmp, stable=True, descending=desc)
        perm = torch.gather(perm, 0, inverse_index)
        print(perm)
    for i in range(len(gpu_cols)):
        gpu_cols[i] = torch.gather(gpu_cols[i], 0, perm)

    elapsed = time.time() - start_time
    print(f"sort on GPU: {elapsed:.2f} seconds")
    torch.cuda.synchronize(device=device)
    return perm

def test_on_cpu_sort(cols): 
    cpu_cols = [c.to('cpu') for c in cols]

    start_time = time.time()
    
    perm = torch.arange(cpu_cols[0].numel(), device=device)
    exchange = vortex.exchange(20_000_000)
    for i, (col, desc) in enumerate(zip(reversed(cpu_cols), [True, True, False, True, False])):
        
        col_d = torch.empty_like(col, device=device)
        exchange.launch([col_d], [col], [torch.tensor([], device=device)], [torch.tensor([])])
        exchange.sync()
        # Sort each column on GPU
        tmp = torch.gather(col_d, 0, perm)
        _, inverse_index = torch.sort(tmp, stable=True, descending=desc)
        perm = torch.gather(perm, 0, inverse_index)
        print(perm)
        del inverse_index
        # Again, force a CUDA sync to ensure sorting completes
    perm_h = torch.empty_like(perm, device='cpu')
    exchange.launch([torch.tensor([], device=device)], [torch.tensor([])], [perm_h], [perm])
    exchange.sync()
    for i in range(len(cpu_cols)):
        cpu_cols[i] = torch.gather(cpu_cols[i], 0, perm_h)

    elapsed = time.time() - start_time
    print(f"sort on CPU: {elapsed:.2f} seconds")
    return perm

def test_naive_sort(src, chunk_size_B = 1_000_000):
    out_sort = []
    out_idx = []
    chunk_size = chunk_size_B // src.element_size()
    for i in range(0, src.shape[0], chunk_size):
        sz = min(i + chunk_size, src.shape[0]) - i
        device_src = src[i :  i + sz].pin_memory().to(torch.device('cuda:0'))
        res_sort, res_idx = torch.sort(device_src)
        out_sort.append(res_sort.to(torch.device('cpu')))
        out_idx.append(res_idx.to(torch.device('cpu')))

    return out_sort, out_idx

def test_vortex_sort_exchange_without_vortex(src, chunk_size = 1_000_000):
    device_mems = get_device_memory_metrics()

    input_stream = torch.cuda.Stream()
    output_stream = torch.cuda.Stream()

    in_host = []
    res_host = [] 
    idx_host = []

    for i in range(0, src.shape[0], chunk_size):
        sz = min(i + chunk_size, src.shape[0]) - i
        in_host.append(src[i :  i + sz])    
        res_host.append(torch.empty((sz,), dtype=torch.int32).pin_memory())
        idx_host.append(torch.empty((sz,), dtype=torch.int32).pin_memory()) 
    
    bufDevice = doubleDeviceBuf(chunk_size, in_host[0].dtype)

    with torch.cuda.stream(input_stream):
        bufDevice.host_to_device_rightsize(in_host[0])
        bufDevice.next.copy_(in_host[0], non_blocking=True)


    for i in range(0, (src.shape[0] + chunk_size - 1) // chunk_size):
        input_stream.synchronize()
        bufDevice.swap()
        if i + 1 < (src.shape[0] + chunk_size - 1) // chunk_size:
            with torch.cuda.stream(input_stream):
                bufDevice.host_to_device_rightsize(in_host[i + 1])
                bufDevice.next.copy_(in_host[i + 1], non_blocking=True)
        res_sort_new, sorted_idx_new = torch.sort(bufDevice.cur) 

        output_stream.synchronize()
        with torch.cuda.stream(output_stream):
            res_sort = res_sort_new.clone()
            res_host[i].copy_(res_sort, non_blocking=True)
            sorted_idx = sorted_idx_new.clone()
            idx_host[i].copy_(sorted_idx, non_blocking=True)
            clean_tensors()

    output_stream.synchronize()

    return res_host, idx_host


def test_vortex_sort_exchange(src, chunk_size = 1_000_000):
    vp = VortexPipeline([src], [torch.int64, torch.int64], torch.sort, chunk_size=chunk_size)
    vp.do_exchange(20_000_000)
    return vp.get_result()

def test_vortex_basic():
    exchange = vortex.exchange(20_000_000)

    dstDevice = [torch.empty((1_000_000_000,), dtype=torch.int32, device='cuda') for _ in range(2)]
    srcHost = [torch.ones((1_000_000_000,), dtype=torch.int32).pin_memory() for _ in range(2)]

    exchange.launch(dstDevice, srcHost, [torch.tensor([])], [torch.tensor([])])
    exchange.sync() # can only be called after you launch!!
    for i in range(2):
        ans = dstDevice[i].to(device = 'cpu')
        assert (ans == srcHost[i]).all()

def test_pipeline_basic():
    src = [torch.randint(1, 100, (100,), dtype=torch.int64).pin_memory()]
    outtype = [torch.int64]
    vp = VortexPipeline(src, outtype, torch.min, chunk_size=10 * 8)
    vp.do_exchange(20_000_000)
    res = vp.get_result()
    for i in range(10):
        assert([torch.min(src[0][i * 10: (i + 1) * 10])] == res[i])

def get_device_memory_metrics():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        mems = []
        for i in range(num_gpus):
            device = torch.device(f"cuda:{i}")
            properties = torch.cuda.get_device_properties(device)
            total_memory_bytes = properties.total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            mems.append(math.floor(0.9 * (total_memory_bytes - reserved_mem)))
        return mems
    else:
        print("CUDA is not available.")

def main():
    # src = torch.randint(1, 100, (tensor_size_B // 8,), dtype=torch.int64).pin_memory()
    test_vortex_basic()
    test_pipeline_basic()
    cols = [
    torch.randint(
        low=0,
        high=(1 << 31),  # just an arbitrary range
        size=(tensor_size_B // 64,),
        dtype=torch.int64,
        device=torch.device('cuda')
    )
    for _ in range(5)
    ]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        r1 = test_on_gpu_sort(cols)

    # print("== test_on_gpu_sort Profile ==")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        r2 = test_on_cpu_sort(cols)
    assert tensor_lists_equal([r1], [r2], device)
    # print("== test_on_gpu_sort Profile ==")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        r3 = test_vortex_sort(cols)
    # print("== test_on_gpu_sort Profile ==")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    assert tensor_lists_equal([r2], [r3], device)

    # start = time.time()
    # res, idx = test_naive_sort(src, chunk_size_B)
    # end = time.time()
    # print(f"Serialized compute and transmission took {end - start:.6f} seconds")
    # # print(res, idx)

    # # for _ in range(1):
    # # start1 = time.time()
    # # res_1, idx_1 = test_vortex_sort_exchange_without_vortex(src, chunk_size)
    # # print(res_1, idx_1)
    # # end1 = time.time()
    # # print(f"test_vortex_sort_exchange_without_vortex took {end1 - start1:.6f} seconds")
    # # assert len(res) == len(res_1) and len(idx) == len(idx_1) and len(res) == len(idx)
    # # assert all(torch.equal(a, b) and torch.equal(c, d) for a, b, c, d in zip(res, res_1, idx, idx_1))

   
    # # Time test_vortex_sort_exchange
    # start2 = time.time()
    # ls_res = test_vortex_sort_exchange(src, chunk_size_B)
    # # print(res_1, idx_1)
    # end2 = time.time()
    # print(f"test_vortex_sort_exchange took {end2 - start2:.6f} seconds")
    # res_1 = [t[0] for t in ls_res]
    # idx_1 = [t[1] for t in ls_res]
    # assert len(res) == len(res_1) and len(idx) == len(idx_1) and len(res) == len(idx) 
    # assert all(torch.equal(a, b) and torch.equal(c, d) for a, b, c, d in zip(res, res_1, idx, idx_1))

    # print(res_1)
    # bounds = find_boundaries_for_merge(res_1, chunk_size)
    # print(bounds)
    # hipTQP.merge(res_1[0].to(torch.int64),res_1[1].to(torch.int64),torch.empty((chunk_size*2,), dtype=torch.int64, device="cuda:0"),
    # idx_1[0].to(torch.int64), idx_1[1].to(torch.int64),torch.empty((chunk_size*2,), dtype=torch.int64, device="cuda:0"))
