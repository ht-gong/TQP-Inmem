import torch
from IO.vortex import set_exchange_to_naive
from IO.vortex_pipeline import VortexPipeline
from utility.logger import perf_logger, set_perf_logger, set_message_logger, set_datasize_logger
from IO.pinned_mem import PinnedMemory, GPUMemory
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
from variable import Variable
import time
import sys 
from pathlib import Path
import TQPlib.tqpmemory as cmp

TOTAL_DATA_SIZE_GB = 20
CHUNK_SIZE_GB = 2
GB = 1_000_000_000

def main():
    alloc = CUDAPluggableAllocator(
        path_to_so_file=f"libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc",
        free_fn_name="pa_cuda_free",
    )
    change_current_allocator(alloc)  # must be called before any CUDA tensors exis

    cmp.init(78*1024**3, 78*1024**3 // 10000)   
    cmp.dump_allocs()
    set_message_logger(enable=True)
    set_perf_logger(enable=True)
    set_datasize_logger()
    set_exchange_to_naive()

    datatype = torch.float32
    mem_pool = PinnedMemory(100, 50)
    gpu_mem_pool = GPUMemory(30, 50)

    numel = TOTAL_DATA_SIZE_GB * GB // 4  # float32 = 4 bytes
    chunk_size = CHUNK_SIZE_GB * GB
    random = torch.rand((numel,), dtype=torch.float32, device='cpu')

    for _ in range(3):
        _, to_sort = mem_pool.malloc(numel, datatype)
        to_sort[:] = random
        sort_input = {1: Variable(to_sort, '')}

        def sorting(args):
            res, ind = torch.sort(args[1].tensor)
            return [res]

        vp = VortexPipeline(sort_input, [datatype], sorting, mem_pool, gpu_mem_pool, name='sort', chunk_size=chunk_size)
        vp.do_exchange(20_000_000)
        res = vp.get_result() 


        mem_pool.free_all()

    for _ in range(3):
        # Custm pipelined implementation
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            chunk_size_elements = chunk_size // 4
            num_chunks = (numel + chunk_size_elements - 1) // chunk_size_elements
            
            # Allocate pinned memory for the input tensor
            _, to_sort = mem_pool.malloc(numel, datatype)
            to_sort[:] = random
            
            # Allocate pinned memory for the output tensor
            _, sorted_tensor = mem_pool.malloc(numel, datatype)

            # Create CUDA streams for H2D, compute, and D2H
            h2d_stream = torch.cuda.Stream()
            compute_stream = torch.cuda.Stream()
            d2h_stream = torch.cuda.Stream()

            # --- Pre-allocate GPU buffers for a 3-stage pipeline ---
            gpu_buffers_in = [
            torch.empty(chunk_size_elements, dtype=datatype, device='cuda'),
            torch.empty(chunk_size_elements, dtype=datatype, device='cuda')
            ]
            gpu_buffers_out = [
            torch.empty(chunk_size_elements, dtype=datatype, device='cuda'),
            torch.empty(chunk_size_elements, dtype=datatype, device='cuda')
            ]

            # Start the pipeline
            start = time.perf_counter()

            # --- Pipeline Main Loop ---
            for i in range(num_chunks + 2):
                # --- Stage 1: H2D (Host-to-Device Transfer) ---
                # Schedule the copy of the *current* chunk `i` to the GPU.
                if i < num_chunks:
                    # Determine the slice of the CPU input tensor
                    start_idx = i * chunk_size_elements
                    end_idx = min(start_idx + chunk_size_elements, numel)
                    chunk_len = end_idx - start_idx
                    
                    # Select the next available GPU buffer from the pool
                    buffer_idx = i % 2
                    
                    # Asynchronously copy the chunk to the GPU on the H2D stream
                    with torch.cuda.stream(h2d_stream):
                        # Use a view to handle the last, possibly smaller, chunk
                        gpu_buffers_in[buffer_idx][:chunk_len].copy_(to_sort[start_idx:end_idx], non_blocking=True)

                # --- Stage 2: Compute (Sorting) ---
                # Schedule the sorting of the *previous* chunk `i-1`.
                compute_chunk_idx = i - 1
                if 0 <= compute_chunk_idx < num_chunks:
                    # Determine which buffer holds the data for this chunk
                    buffer_idx = compute_chunk_idx % 2
                    
                    # Define the size of the chunk we are computing
                    start_idx = compute_chunk_idx * chunk_size_elements
                    chunk_len = min(chunk_size_elements, numel - start_idx)

                    # The compute stream must wait for the H2D stream to finish copying this chunk
                    
                    with torch.cuda.stream(compute_stream):
                        in_view = gpu_buffers_in[buffer_idx][:chunk_len]
                        out_view = gpu_buffers_out[buffer_idx][:chunk_len]
                        
                        # Perform the sort
                        out_view[:], _ = torch.sort(in_view)
                        # out_view[:] = in_view[:]

                # --- Stage 3: D2H (Device-to-Host Transfer) ---
                # Schedule the copy of the sorted *second-to-last* chunk `i-2` back to the CPU.
                d2h_chunk_idx = i - 2
                if 0 <= d2h_chunk_idx < num_chunks:
                    # Determine which buffer holds the sorted data for this chunk
                    buffer_idx = d2h_chunk_idx % 2
                    
                    # Define the slice of the CPU output tensor
                    start_idx = d2h_chunk_idx * chunk_size_elements
                    end_idx = min(start_idx + chunk_size_elements, numel)
                    chunk_len = end_idx - start_idx
                    
                    # The D2H stream must wait for the compute stream to finish sorting this chunk
                    
                    with torch.cuda.stream(d2h_stream):
                        # Asynchronously copy the sorted chunk back to the CPU on the D2H stream
                        sorted_tensor[start_idx:end_idx].copy_(gpu_buffers_out[buffer_idx][:chunk_len], non_blocking=True)
                
                h2d_stream.synchronize()
                compute_stream.synchronize()
                d2h_stream.synchronize()

            # Synchronize all streams to ensure all operations are complete before stopping the timer
            torch.cuda.synchronize()
            end = time.perf_counter()
            print(f"Custom pipeline time = {end - start:.4f} seconds")

            # --- NOTE: The functional correctness issue remains. `res` is not globally sorted. ---
            # To get a final sorted result, a merge step is required here.
            res = sorted_tensor

            mem_pool.free_all()

        # prof.export_chrome_trace('./debug_trace.json')
