import torch
import time
import os
import constants
from variable import Variable
from utility.logger import perf_logger, message_logger

def tqp_scan(gpu_enable, o_tensor_group, args, config_json, schema_table, SF, mem_pool, name="scan"):
  tensor_path = config_json.get('tensors path')
  type_mapping = {
    "identifier": "int",
    "integer": "int",
    "decimal": "float",
    "date": "date",
    "variable text": "string",
    "fixed text": "string"
  }
  col_names, col_ids = args['output names'], args['output ids'] 
  rows = 0

  for col_name, col_id in zip(col_names, col_ids):
    col_name = col_name.upper()
    with perf_logger().time(f"{name} read from disk"):
      path = os.path.join(tensor_path, f'SF{SF}-tensor-{col_name}.pth')
      o_tensor_group[col_id] = Variable(list(torch.jit.load(path).parameters())[0],
                                       type_mapping[schema_table[col_name].split(',')[0]])
      if col_name == schema_table['PRIMARY_KEY']:
        o_tensor_group[col_id].is_sorted = True

      message_logger().info("%s %s", col_name, o_tensor_group[col_id])

      rows = o_tensor_group[col_id].tensor.shape[0]
      if o_tensor_group[col_id].tensor_type == 'date':
        o_tensor_group[col_id].tensor = o_tensor_group[col_id].tensor.to(constants.date_dtype)
      elif torch.is_floating_point(o_tensor_group[col_id].tensor):
        o_tensor_group[col_id].tensor = o_tensor_group[col_id].tensor.to(constants.float_dtype)
      elif o_tensor_group[col_id].tensor_type == 'string':
        o_tensor_group[col_id].tensor = o_tensor_group[col_id].tensor.to(constants.string_dtype)
      else:
        assert o_tensor_group[col_id].tensor.dim() == 1
        o_tensor_group[col_id].tensor = o_tensor_group[col_id].tensor.to(constants.int_dtype)
      

    if gpu_enable:
      with perf_logger().time(f"{name} pin memory"):
        orig_data = o_tensor_group[col_id].tensor
        ptr, o_tensor_group[col_id].tensor = mem_pool.malloc_like(o_tensor_group[col_id].tensor)
        o_tensor_group[col_id].tensor[:] = orig_data.to('cuda')
        o_tensor_group[col_id].backing_mem_ptr = ptr

  if gpu_enable:
    torch.cuda.synchronize()
  
  return rows