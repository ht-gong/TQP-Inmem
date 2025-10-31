import torch
import itertools
from utility.tensor_utils import resize_tensor_list, show_tensor_usage, allocate_tensor_list
from typing import List, Dict
import IO.vortex as vortex
from utility.logger import perf_logger, message_logger, datasize_logger
from enum import Enum
from utility.tensor_utils import dtype_size
from variable import Variable, VariableState
from dataclasses import dataclass
from conversion import index_with_null
from IO.cache_manager import zero_copy_mask_policy, zero_copy_rearrange_policy
import TQPlib.tqpmemory as cmp

@dataclass
class ColumnMetaData:
    memory_ptr: int 
    tensor: torch.Tensor
    cur_ind: int 
    placement: VariableState


class PipeStep(str, Enum):
    PIN_INPUT = "pinning input"
    PIN_OUTPUT = "pinning output"
    RESIZE = "resize"
    APPLY_OP = "apply op"
    SYNC = "transfer sync"
    ELAPSED = "elapsed"

    def __str__(self):
        return self.value

class doubleDeviceBuf:
    def __init__(self, target_tensors, device='cuda:0'):
        if len(target_tensors) == 0:
            raise ValueError("target tensors are empty")
        self.cur = [torch.tensor([], dtype=t.dtype, device=device) for t in target_tensors] 
        self.next = [torch.tensor([], dtype=t.dtype, device=device) for t in target_tensors] 

    def allocate_shape_cur(self, target_tensors):
        # We need this because the destination device tensor may not have the right size
        if len(self.cur) != len(target_tensors):
            raise ValueError("target_tensors count do not match doublebuf")
        resize_tensor_list(self.cur, target_tensors)

    def allocate_shape_next(self, target_tensors):
        # We need this because the destination device tensor may not have the right size
        if len(self.next) != len(target_tensors):
            raise ValueError("target_tensors count do not match doublebuf")
        resize_tensor_list(self.next, target_tensors)
                  
    def swap(self):
        self.cur, self.next = self.next, self.cur

def handle_GPU_transfers(gpu_copy_stream: torch.cuda.Stream,
                        dst_device: List[torch.Tensor], 
                        src_host: List[torch.Tensor], 
                        placement_src: List[VariableState],
                        dst_host: List[torch.Tensor], 
                        src_device: List[torch.Tensor],
                        placement_dst: List[VariableState], 
                        **zero_copy_policies):
    
    assert isinstance(dst_device, list) and isinstance(src_host, list) and len(dst_device) == len(src_host), "Invalid host-to-device arguments."
    assert isinstance(dst_host, list) and isinstance(src_device, list) and len(dst_host) == len(src_device), "Invalid device-to-host arguments."
    assert isinstance(placement_src, list) and isinstance(placement_dst, list) and \
          len(placement_src) == len(src_host) and len(placement_dst) == len(dst_host),"Invalid placement arguments."
    for z in zero_copy_policies.values():
        assert isinstance(z, list) and len(z) == len(src_host), "Invalid zero-copy arguments"
    
    exchange_dst_d, exchange_src_h, exchange_dst_h, exchange_src_d = [], [], [], []
    filtered_zero_copy_policies = {k: [] for k in zero_copy_policies}
    with torch.cuda.stream(gpu_copy_stream):
        for i in range(len(src_host)):
            if placement_src[i] == VariableState.GPU:
                assert dst_device[i].device == src_host[i].device
                dst_device[i] = src_host[i] # no copy 
            else:
                exchange_dst_d.append(dst_device[i])
                exchange_src_h.append(src_host[i])
                for j in zero_copy_policies:
                    filtered_zero_copy_policies[j].append(zero_copy_policies[j][i])

        for i in range(len(dst_host)):
            if placement_dst[i] == VariableState.GPU:
                datasize_logger().record('Naive GPU Out', src_device[i].numel() * src_device[i].element_size())
                dst_host[i].copy_(src_device[i], non_blocking=True) # copy for easier memory management
            else:
                exchange_dst_h.append(dst_host[i])
                exchange_src_d.append(src_device[i])
    return (exchange_dst_d, exchange_src_h, exchange_dst_h, exchange_src_d), filtered_zero_copy_policies

class InMemoryPipeline:
    def __init__(self, input_columns: Dict[int, Variable], output_columns_type: List[torch.dtype], operator,\
                 cpu_mem_pool, gpu_mem_pool=None, div=None, chunk_size=20_000_000_000, pass_mask=False, mask_map: Dict[int, int] = {}, name="VortexPipe"):

        self.__input_cols = input_columns
        self.__out_host = []
        self.__output_columns_typed = [torch.tensor([], dtype=t, device='cuda:0') for t in output_columns_type] 
        self.__operator = operator
        self.__name = name
        self.__cpu_mem_pool = cpu_mem_pool
        self.__gpu_mem_pool = gpu_mem_pool
        self.__pass_mask = pass_mask

        self.__mask_map = {}
        cols = list(self.__input_cols.keys())
        for k, v in mask_map.items():
            assert v in cols, "mask col not passed in"
            self.__mask_map[cols.index(k)] = cols.index(v)
    
        for col in cols:
            assert self.__input_cols[col].tensor.is_cuda
            self.__input_cols[col].tensor = self.__input_cols[col].tensor.contiguous()

    def do_exchange(self, granularity: int):
        perf_logger().start(f"{self.__name}")
        outbuf = self.safe_apply_op([t.tensor for t in self.__input_cols.values()])
        perf_logger().stop(f"{self.__name}")
        torch.cuda.synchronize()
        for i in range(len(outbuf)):
            ptr, storage = self.__gpu_mem_pool.malloc_like(outbuf[i])
            storage[:] = outbuf[i]
            self.__out_host.append(ColumnMetaData(ptr, storage, -1, VariableState.GPU))
            outbuf[i] = None 
        
 
    def safe_apply_op(self, input_tensors: List[torch.tensor]):
        if all(t.numel() == 0 for t in input_tensors):
            return [torch.tensor([], device='cuda') for _ in self.__output_columns_typed]
        # Reconstruct input group for operator computation
        input_dict = {}
        for (col_name, old_var), (i, tensor) in zip(self.__input_cols.items(), enumerate(input_tensors)):
            if i not in self.__mask_map.values():
                # Only process actual columns
                if i in self.__mask_map.keys():
                    # Has not went through zero-copy 
                    input_dict[col_name] = Variable(tensor[input_tensors[self.__mask_map[i]]],
                                                        old_var.tensor_type)
                else:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type) 
            else: 
                if self.__pass_mask:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type)
                else:
                    input_tensors[i].tensor = None # Clear mask tensor
        res = self.__operator(input_dict)
        if isinstance(res, torch.Tensor) and res.ndim == 0:
            res = [res]
        return list(res)

    def get_result(self):
        return self.__out_host


class VortexPipeline:
    """
    Creates a Vortex Pipeline that operates on input_columns in batches using a specified operator,
    either using a pre-specified division or a fixed chunk_size, 
    then the results are returned in batches back to the output_columns.
    REQUIRES: Both the input_columns (as Variables) and the output_columns are allocated on the host (CPU). 
              The output types are known apriori.
    """
    ESTIMATION_CONSTANT = 2
    def __init__(self, input_columns: Dict[int, Variable], output_columns_type: List[torch.dtype], operator,\
                 cpu_mem_pool, gpu_mem_pool=None, div=None, chunk_size=20_000_000_000, pass_mask=False, mask_map: Dict[int, int] = {}, name="VortexPipe"):

        self.__input_cols = input_columns
        self.__out_host = []
        self.__output_columns_typed = [torch.tensor([], dtype=t, device='cuda:0') for t in output_columns_type] 
        self.__operator = operator
        self.__name = name
        self.__cpu_mem_pool = cpu_mem_pool
        self.__gpu_mem_pool = gpu_mem_pool
        self.__pass_mask = pass_mask

        # placement for tensors
        self.__input_placement = [col.state for col in input_columns.values()]
        self.__output_placement = [VariableState.CPU for _ in output_columns_type] # Default placeholder
        
        self.__mask_map = {}
        cols = list(self.__input_cols.keys())
        for k, v in mask_map.items():
            assert v in cols, "mask col not passed in"
            self.__mask_map[cols.index(k)] = cols.index(v)

    
        for col in input_columns.values():
            col.tensor = col.tensor.contiguous()
            if col.state == VariableState.CPU:
                assert col.tensor.is_pinned()

        in_chunks_host = []
        arr_size = next(iter(input_columns.values())).tensor.shape[0]

        if not div: 
            if chunk_size < 0: # -chunk_size = number of rows
                elem_perchunk = -chunk_size
            else:
                elem_size = sum([t.tensor.element_size() * (1 if t.tensor.dim() == 1 else t.tensor.size(1)) for t in input_columns.values()])
                elem_perchunk = chunk_size // elem_size

            chunks = (arr_size + elem_perchunk - 1) // elem_perchunk
            self.__stages = chunks + 2 # 2 Extra stages in prologue and prelogue

            for i in range(self.__stages):
                sz = min(elem_perchunk, max(arr_size - i * elem_perchunk, 0)) 
                if i < self.__stages - 2:
                    in_chunks_host.append(sz)
                else:
                    in_chunks_host.append(0)
            in_chunks_host = [list(itertools.accumulate(in_chunks_host)) for _ in range(len(input_columns))]
        
        else:
            self.__stages = len(div[0]) + 2
            in_chunks_host = [d + [d[-1], d[-1]] for d in div]

        self.__in_host = [[] for _ in range(self.__stages)]
        for i in range(self.__stages):
            for j, col in enumerate(input_columns.values()):
                if i == 0:
                    self.__in_host[i].append(col.tensor[0:in_chunks_host[j][i]])
                else:
                    self.__in_host[i].append(col.tensor[in_chunks_host[j][i - 1]:in_chunks_host[j][i]])

    def do_exchange(self, granularity: int):
        perf_logger().start(f"{self.__name}")

        compute_stream = torch.cuda.Stream() 
        gpu_copy_stream = torch.cuda.Stream(device="cuda:0") 
        exchange = vortex.exchange(granularity)

        ### Precompute for zero-copy mask policy
        masking_info = []
        with torch.cuda.stream(compute_stream):
            for stage in self.__in_host:
                stage_mask_info = [(None, None) for _ in range(len(stage))]
                for data_col, mask_col in self.__mask_map.items():
                    if zero_copy_mask_policy(stage[mask_col], stage[data_col]):
                        stage_mask_info[data_col] = (stage[mask_col], stage[mask_col].sum().item())
                masking_info.append(stage_mask_info)
        compute_stream.synchronize()
                    
        # print(self.__stges, masking_info)

        with perf_logger().time(f"{self.__name} {PipeStep.RESIZE}"):
            in_buf_device = doubleDeviceBuf(self.__in_host[0])
            in_buf_device.allocate_shape_cur(self.__in_host[0])
        
        transfer_params, zero_copy_policies = handle_GPU_transfers(gpu_copy_stream, in_buf_device.cur, self.__in_host[0], self.__input_placement, 
                                              [], [], [], mask_info=masking_info[0])
        exchange.launch(*transfer_params, **zero_copy_policies)
 
        with perf_logger().time(f'{self.__name} {PipeStep.SYNC}', sum(t.nbytes for t in self.__in_host[0])):
            exchange.sync() 
            gpu_copy_stream.synchronize()
            
        in_buf_device.allocate_shape_next(self.__in_host[1])
        out_buf_device = doubleDeviceBuf(self.__output_columns_typed)
        
        transfer_params, zero_copy_policies = handle_GPU_transfers(gpu_copy_stream, in_buf_device.next, self.__in_host[1], self.__input_placement, 
                                              [], [], [], mask_info=masking_info[1])
        exchange.launch(*transfer_params, **zero_copy_policies) 
        
        with perf_logger().time(f"{self.__name} {PipeStep.APPLY_OP}", sum(t.nbytes for t in in_buf_device.cur)):
            with torch.cuda.stream(compute_stream):
                out_buf_device.cur = self.safe_apply_op(in_buf_device.cur, masking_info[0])

                #### Use first output chunk info to estimate data size for GPU/CPU placement
                self.__output_placement = []
                self.__output_dimensions = []
                for out_buf in out_buf_device.cur:
                    self.__output_dimensions.append(out_buf.shape[1:])
                    estimated_numel = out_buf.numel() * self.ESTIMATION_CONSTANT * (self.__stages - 2)
                    estimated_size_b = estimated_numel * dtype_size(out_buf.dtype)
                    if self.__gpu_mem_pool and self.__gpu_mem_pool.free_space() > estimated_size_b:
                        self.__output_placement.append(VariableState.GPU)
                        ptr, storage = self.__gpu_mem_pool.malloc(estimated_numel, out_buf.dtype)
                    else:
                        self.__output_placement.append(VariableState.CPU)
                        ptr, storage = self.__cpu_mem_pool.malloc(estimated_numel, out_buf.dtype)
                    self.__out_host.append(ColumnMetaData(ptr, storage, 0, self.__output_placement[-1]))
     

            compute_stream.synchronize()

        with perf_logger().time(f'{self.__name} {PipeStep.SYNC}', sum(t.nbytes for t in self.__in_host[1])):
            exchange.sync()
            gpu_copy_stream.synchronize()

        for i in range(2, self.__stages):
            in_buf_device.swap()
            in_buf_device.allocate_shape_next(self.__in_host[i])
            out_buf_device.next = None

            with torch.cuda.stream(compute_stream):
                out_stage = []
                for j in range(len(out_buf_device.cur)):
                    index = self.__out_host[j].cur_ind
                    out_stage.append(self.__out_host[j].tensor[index: index + out_buf_device.cur[j].numel()]
                                                       .view(out_buf_device.cur[j].shape))
                    self.__out_host[j].cur_ind = index + out_buf_device.cur[j].numel()
            compute_stream.synchronize()

            transfer_params, zero_copy_policies = handle_GPU_transfers(gpu_copy_stream, in_buf_device.next, self.__in_host[i], self.__input_placement, 
                                                 out_stage, out_buf_device.cur, self.__output_placement, mask_info=masking_info[i])
            exchange.launch(*transfer_params, **zero_copy_policies) 

            with perf_logger().time(f"{self.__name} {PipeStep.APPLY_OP}", sum(t.nbytes for t in in_buf_device.cur)):
                with torch.cuda.stream(compute_stream):
                    out_buf_device.next = self.safe_apply_op(in_buf_device.cur, masking_info[i-1])

                compute_stream.synchronize()
            
            with perf_logger().time(f"{self.__name} {PipeStep.SYNC}", sum(t.nbytes for t in self.__in_host[i])):
                exchange.sync()
                gpu_copy_stream.synchronize()
            
            out_buf_device.swap()
        
        perf_logger().stop(f"{self.__name}")
 
    def safe_apply_op(self, input_tensors: List[torch.tensor], mask_info):
        if all(t.numel() == 0 for t in input_tensors):
            return [torch.tensor([], device='cuda') for _ in self.__output_columns_typed]
        # Reconstruct input group for operator computation
        input_dict = {}
        for (col_name, old_var), (i, tensor) in zip(self.__input_cols.items(), enumerate(input_tensors)):
            if i not in self.__mask_map.values():
                # Only process actual columns
                if i in self.__mask_map.keys() and mask_info[i][0] is None:
                    # Has not went through zero-copy 
                    input_dict[col_name] = Variable(tensor[input_tensors[self.__mask_map[i]]],
                                                        old_var.tensor_type)
                else:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type) 
            else: 
                if self.__pass_mask:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type)
                else:
                    input_tensors[i].tensor = None # Clear mask tensor

        res = self.__operator(input_dict)
        if isinstance(res, torch.Tensor) and res.ndim == 0:
            res = [res]
        return list(res)

    def get_result(self):
        for dim, col in zip(self.__output_dimensions, self.__out_host):
            #### Release overestimated & unused memory
            col.tensor = col.tensor[:col.cur_ind].view(-1, *dim)
            if col.placement == VariableState.GPU:
                self.__gpu_mem_pool.partial_free(col.memory_ptr, col.cur_ind * dtype_size(col.tensor.dtype))
            else:
                self.__cpu_mem_pool.partial_free(col.memory_ptr, col.cur_ind * dtype_size(col.tensor.dtype))
        return self.__out_host

from typing import Dict, List, Optional

def rearrange_pipeline(
    tensors: Dict[int, Variable],
    outIdx: Variable,
    cpu_mem_pool,
    gpu_mem_pool,
    type,
    name: str = "rearrange pipe",
):
    """
    Rearrange a dict of column Variables by 'outIdx', stage-by-stage, using a GPU copy stream
    and Vortex async exchange. For each column:
      1) decide zero-copy policy,
      2) stage into a GPU buffer,
      3) (optionally) apply index/select on GPU,
      4) write results back into pooled storage (GPU if enough space, else CPU).
    """
    perf_logger().start(name)

    # ---- Early exit
    if not tensors:
        message_logger().debug("No cols for rearrange")
        perf_logger().stop(name)
        return

    # ---- Setup
    n = len(tensors)
    gpu_copy_stream = torch.cuda.Stream(device="cuda:0")
    exchange = vortex.exchange()

    # Keep original order of dict iteration (Py3.7+ preserves insertion order)
    col_buffer: List[Variable] = []
    for k, var in tensors.items():
        # Ensure contiguity (fix: .contiguous() is NOT in-place)
        if not var.tensor.is_contiguous():
            var.tensor = var.tensor.contiguous()

        # If on CPU, require pinned memory
        if var.state == VariableState.CPU:
            assert var.tensor.is_pinned(), f"col {k} is not pinned"

        # message_logger().debug("Rearrange set shape: %s", var.tensor.shape)
        col_buffer.append(var)

    # Logging selectivity info
    message_logger().debug(
        "out Idx size: %s, out Idx selectivity: %s",
        outIdx.tensor.shape,
        outIdx.tensor.shape[0] / col_buffer[0].tensor.shape[0] if col_buffer[0].tensor.shape[0] != 0 else 0,
    )

    # ---- Selection function
    select_function = (index_with_null if type == "right-outer"
                       else (lambda x, idx: x[idx]))

    # ---- Helper: ensure a Variable is on GPU; return its tensor on GPU
    def ensure_gpu(variable: Optional[Variable]) -> Optional[torch.Tensor]:
        if variable is None:
            return None
        if variable.state == VariableState.CPU:
            tmp = torch.empty_like(variable.tensor, device="cuda")
            exchange.launch([tmp], [variable.tensor],
                            [torch.tensor([])], [torch.tensor([], device="cuda")])
            with perf_logger().time(f"{name} {PipeStep.SYNC}", variable.tensor.nbytes):
                exchange.sync()
            return tmp
        return variable.tensor

    # Materialize outIdx on GPU and validate shape
    out_idx_t = ensure_gpu(outIdx)
    assert out_idx_t is not None and out_idx_t.ndim == 1, "OutIdx dimension must be 1-D"

    # ---- Precompute zero-copy rearrange policy per stage
    rearrange_info: List[List[Optional[torch.Tensor]]] = []
    with torch.cuda.stream(gpu_copy_stream):
        for i in range(n):
            if zero_copy_rearrange_policy(out_idx_t, col_buffer[i].tensor):
                rearrange_info.append([out_idx_t])
            else:
                rearrange_info.append([None])
    gpu_copy_stream.synchronize()

    # ---- GPU stage buffers (each as a single-element list to match handle_GPU_transfers API)
    gpu_buffer: List[Optional[List[torch.Tensor]]] = [None] * n

    # ---- Small helper to allocate stage buffer
    def _alloc_stage_buf(i: int) -> List[torch.Tensor]:
        src = col_buffer[i].tensor
        if rearrange_info[i][0] is not None:
            # Directly allocate the gathered shape
            return [torch.empty((out_idx_t.shape[0], *src.shape[1:]),
                               dtype=src.dtype, device="cuda")]
        else:
            # Mirror source shape first, then we'll index on GPU
            return [torch.empty_like(src, device="cuda")]

    # ---- Small helper to transfer for a stage
    def _stage_transfer(i: int):
        prev_src = [col_buffer[i-1].tensor] if i > 0 else []
        prev_dst = gpu_buffer[i-1] if i > 0 else []
        prev_states = [col_buffer[i-1].state] if i > 0 else []

        gpu_buffer[i] = _alloc_stage_buf(i)

        transfer_params, zero_copy_policies = handle_GPU_transfers(
            gpu_copy_stream,
            gpu_buffer[i],
            [col_buffer[i].tensor],
            [col_buffer[i].state],
            prev_src,
            prev_dst if prev_dst is not None else [],
            prev_states,
            rearrange_info=rearrange_info[i]
        )
        exchange.launch(*transfer_params, **zero_copy_policies)

        with perf_logger().time(f"{name} {PipeStep.SYNC}", col_buffer[i].tensor.nbytes):
            exchange.sync()
            gpu_copy_stream.synchronize()

        # If not zero-copy, apply the index on the staged tensor
        if rearrange_info[i][0] is None:
            with perf_logger().time(f"{name} {PipeStep.APPLY_OP}", gpu_buffer[i][0].nbytes):
                with torch.cuda.stream(gpu_copy_stream):
                    gpu_buffer[i] = [select_function(gpu_buffer[i][0], out_idx_t)]
                gpu_copy_stream.synchronize()

    # ---- Stage 0
    _stage_transfer(0)

    # ---- Recycle previous column storage each step & run subsequent stages
    for i in range(1, n):
        # Decide where to place previous column's final storage (GPU if enough space)
        bytes_needed = out_idx_t.numel() * dtype_size(gpu_buffer[i-1][0].dtype)
        if gpu_mem_pool and gpu_mem_pool.free_space() > bytes_needed:
            ptr, storage = gpu_mem_pool.malloc_like(gpu_buffer[i-1][0])
            target_state = VariableState.GPU
        else:
            ptr, storage = cpu_mem_pool.malloc_like(gpu_buffer[i-1][0])
            target_state = VariableState.CPU

        # Replace previous column with pooled storage and free old backing
        col_buffer[i-1].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=gpu_mem_pool)
        col_buffer[i-1] = Variable(storage, col_buffer[i-1].tensor_type, ptr, target_state)

        # Transfer stage i using previous stage as pipeline source
        _stage_transfer(i)

        # Drop previous GPU staging tensor to release memory
        gpu_buffer[i-1][0] = None

    # ---- Final write-back for last stage into pooled storage
    bytes_needed = out_idx_t.numel() * dtype_size(gpu_buffer[n-1][0].dtype)
    if gpu_mem_pool and gpu_mem_pool.free_space() > bytes_needed:
        ptr, storage = gpu_mem_pool.malloc_like(gpu_buffer[n-1][0])
        final_state = VariableState.GPU
    else:
        ptr, storage = cpu_mem_pool.malloc_like(gpu_buffer[n-1][0])
        final_state = VariableState.CPU

    col_buffer[n-1].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=gpu_mem_pool)
    col_buffer[n-1] = Variable(storage, col_buffer[n-1].tensor_type, ptr, final_state)

    transfer_params, zero_copy_policies = handle_GPU_transfers(
        gpu_copy_stream,
        [], [], [],
        [col_buffer[n-1].tensor],
        gpu_buffer[n-1],
        [col_buffer[n-1].state],
    )
    exchange.launch(*transfer_params, **zero_copy_policies)

    with perf_logger().time(f"{name} {PipeStep.SYNC}", col_buffer[n-1].tensor.nbytes):
        exchange.sync()
        gpu_copy_stream.synchronize()

    for i, k in enumerate(tensors):
        tensors[k] = col_buffer[i]

    perf_logger().stop(name)

