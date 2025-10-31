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
import operators.aggregate
import parsing 
from variable import Variable, torch_to_type, type_to_torch, VariableState

TOTAL_DATA_SIZE_GB = 4
CHUNK_SIZE_GB = 10
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
    cpu_mem_pool = PinnedMemory(60, 50)
    gpu_mem_pool = GPUMemory(30, 50)

    bytes_per_row = 24
    num_rows = TOTAL_DATA_SIZE_GB * GB  // bytes_per_row

    chunk_size = CHUNK_SIZE_GB * GB

    if False:
        l_quantity = Variable(torch.rand((num_rows,), dtype=torch.float64, device='cpu').pin_memory(), '')
        l_extendedprice = Variable(torch.randint(0, 99999, (num_rows,), dtype=torch.int64, device='cpu').pin_memory(), '')
        l_linestatus = Variable(torch.randint(0, 2, (num_rows,), dtype=torch.int32, device='cpu').pin_memory(), '')
        l_returnflag = Variable(torch.randint(0, 2, (num_rows,), dtype=torch.int32, device='cpu').pin_memory(), '')
    else:
        l_quantity = Variable(torch.rand((num_rows,), dtype=torch.float64, device='cuda'), '', var_state=VariableState.GPU)
        l_extendedprice = Variable(torch.randint(0, 99999, (num_rows,), dtype=torch.int64, device='cuda'), '', var_state=VariableState.GPU)
        l_linestatus = Variable(torch.randint(0, 2, (num_rows,), dtype=torch.int32, device='cuda'), '',  var_state=VariableState.GPU)
        l_returnflag = Variable(torch.randint(0, 2, (num_rows,), dtype=torch.int32, device='cuda'), '',  var_state=VariableState.GPU)

    i_tensor_group = {
        100: l_returnflag,
        101: l_linestatus,
        96: l_quantity,
        97: l_extendedprice
    }

    args = {'name': 'Aggregate', 
            'Input': ['l_returnflag#100', 'l_linestatus#101', 'sum#149'], 
            'Aggregate Attributes': ['sum(l_quantity#96)#131', 'sum(l_extendedprice#97)#132'], 'is partial': False, 
            'keys': ['l_returnflag#100', 'l_linestatus#101'], 
            'results': [['l_returnflag#100'], ['l_linestatus#101'], [[[['sum', ['l_quantity#96']], '#', '131'], 'AS', 'sum_qty#122']], [[[['sum', ['l_extendedprice#97']], '#', '132'], 'AS', 'sum_base_price#123']]], 
            'input ids': [96, 97, 100, 101]}

    transfer_set = [96, 97, 100, 101]
  
    transfer_types = []
    for i in transfer_set:
        transfer_types.extend(i_tensor_group[i].get_type_torch())

    key_ids, key_ids_new, key_types, key_types_name, key_width = [], [], [], [], []
    agg_ids, agg_types = [], []
    agg_ops = []

    for tree in args['results']:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
        if isinstance(tree, str) or (len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)): # Key
            key_ids.append(operators.aggregate.get_id(tree if isinstance(tree, str) else tree[0]))
            key_ids_new.append(operators.aggregate.get_id(tree if isinstance(tree, str) else tree[-1])) # MAY REASSIGN ID
            key_types.extend(i_tensor_group[key_ids[-1]].get_type_torch())
            key_types_name.append(i_tensor_group[key_ids[-1]].tensor_type)
            key_width.append(1 if i_tensor_group[key_ids[-1]].tensor.dim() == 1 else i_tensor_group[key_ids[-1]].tensor.size(1))
        else:
            agg_ids.append(operators.aggregate.get_id(parsing.right_most(tree)))
            agg_types.append(operators.aggregate.get_result_type(tree, transfer_set, transfer_types))
            agg_ops.append(operators.aggregate.get_ops(tree))
    
    for i in range(1, len(key_width)):
        key_width[i] += key_width[i-1]

    def agg_wrapper(columns):
        cnt_rows = next(iter(columns.values())).tensor.size(0)

        device_out_tensor_group = {}

        if not args['keys']:
          # Expected dtype int64 for index
          unique_key_cols, inverted_index = torch.tensor([[1]], device='cuda'),\
                                            torch.zeros(cnt_rows, dtype=torch.int64, device='cuda')
          
        else:
          key_ids_d = [operators.aggregate.get_id(key) for key in args['keys']]
          key_sizes = []

          # Stack keys columns together into a 2D tensor
          key_cols = None

          for key_id in key_ids_d:
            if columns[key_id].tensor.dim() < 2:
              tmp_col = columns[key_id].tensor.unsqueeze(1)
              key_sizes.append(1)
            else:
              tmp_col = columns[key_id].tensor
              key_sizes.append(tmp_col.shape[-1])

            key_cols = tmp_col if key_cols is None else torch.cat((key_cols, tmp_col), dim=1)
            if len(key_sizes) >= 2:
              key_sizes[-1] += key_sizes[-2]

          # New Logic: Hash and then unique
          # enable_hash_aggregate = key_cols.size(1) >= 3
          enable_hash_aggregate = True
          if enable_hash_aggregate:
            hash_coeff = torch.randint(2, 10000000, (key_cols.size(1),), dtype=torch.int64, device='cuda')
            if key_cols.dim() == 1:
              key_cols = key_cols.unsqueeze(1)
            hash_tensor = (key_cols * hash_coeff).sum(dim=1)

            unique_key_cols, inverted_index = torch.unique(hash_tensor, dim=0, return_inverse=True)
            unique_key_cols = torch.zeros((unique_key_cols.size(0), key_cols.size(1)), dtype=key_cols.dtype, device='cuda')
            unique_key_cols.scatter_(0, inverted_index.unsqueeze(1).expand(-1, key_cols.size(1)), key_cols)
          else:
            unique_key_cols, inverted_index = torch.unique(key_cols, dim=0, return_inverse=True)

          del tmp_col, key_cols

        cnt_groups = unique_key_cols.shape[0]


        for tree in args['results']:
          while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
          
          agg_id = operators.aggregate.get_id(parsing.right_most(tree))
          if isinstance(tree, str) or (len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)): # Key
            key_id = operators.aggregate.get_id(tree if isinstance(tree, str) else tree[0])
            pos = key_ids_d.index(key_id)
            pos_l = 0 if pos == 0 else key_sizes[pos-1]
            pos_r = key_sizes[pos]
            device_out_tensor_group[agg_id] = unique_key_cols[:, pos_l:pos_r]
            device_out_tensor_group[agg_id] = device_out_tensor_group[agg_id].to(columns[key_id].tensor.dtype)
          else:
            # Aggregation
            device_out_tensor_group[agg_id] = operators.aggregate.evaluate('cuda', columns, tree[0], cnt_rows, cnt_groups, inverted_index)

        for col in list(columns.keys()):
          del columns[col]

        res = []
        for i in key_ids_new:
            res.append(device_out_tensor_group[i])
        for i in agg_ids:
            res.append(device_out_tensor_group[i])
        
        for i in range(len(res)):
            res[i] = res[i].contiguous()
        return res
    

    for _ in range(3):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
          agg_pipeline = VortexPipeline(
              {i: i_tensor_group[i] for i in transfer_set}, 
              key_types + agg_types,
              agg_wrapper, 
              gpu_mem_pool=gpu_mem_pool,
              cpu_mem_pool=cpu_mem_pool,
              chunk_size=chunk_size, 
              name=f"aggregation pipe") 
          
        agg_pipeline.do_exchange(20_000_000)
        result = agg_pipeline.get_result()
          
        o_tensor_group = {}
        rows_keys = [result[i].tensor for i in range(len(key_types))]
        rows_vals = [result[i].tensor for i in range(len(key_types), len(key_types) + len(agg_types))]
        if len(rows_keys) > 0:
            rows_keys = [key.unsqueeze(1) if key.dim() == 1 else key for key in rows_keys]
            rows_keys = torch.cat(rows_keys, dim=1)

            rows_keys = rows_keys.to('cuda')
            groups, inv = torch.unique(rows_keys, dim=0, return_inverse=True)
            _, groups_cpu = cpu_mem_pool.malloc_like(groups)
            groups_cpu.copy_(groups)

            # groups = groups.to('cpu')
            # inv = inv.to('cpu')

            for i in range(len(rows_vals)):
                if agg_ops[i] == 'avg':
                    agg_ops[i] = 'mean'
                if agg_ops[i] == 'count':
                    agg_ops[i] = 'sum'

                res = torch.zeros(groups.size(0), dtype=rows_vals[i].dtype, device='cuda')
              # res = torch.zeros(groups.size(0), dtype=rows_vals[i].dtype, device='cuda') # TODO: use vortex
                rows_vals[i] = rows_vals[i].to('cuda')
                res.scatter_reduce_(0, inv, rows_vals[i], reduce=agg_ops[i], include_self=False)
                _, res_cpu = cpu_mem_pool.malloc_like(res)
                res_cpu.copy_(res)
                  # res = res.to('cpu')
                o_tensor_group[agg_ids[i]] = Variable(res_cpu, torch_to_type(res_cpu.dtype))
            
            for i in range(len(key_width)):
                group_chunk = groups_cpu[:, (key_width[i-1] if i > 0 else 0) : key_width[i]].to(type_to_torch(key_types_name[i]))
                ptr, storage = cpu_mem_pool.malloc_like(group_chunk)
                storage.copy_(group_chunk)
                o_tensor_group[key_ids_new[i]] = Variable(storage,
                                                    key_types_name[i])
        else:
            for i in range(len(rows_vals)):
                ptr, storage = cpu_mem_pool.malloc_like(rows_vals[i])
                storage[:] = rows_vals[i]
                o_tensor_group[agg_ids[i]] = Variable(storage, torch_to_type(rows_vals[i].dtype))
      

