import time
import torch
import pandas as pd
from conversion import num_to_str, float_to_date
from parsing import get_id, get_name
from utility.logger import message_logger

def tqp_output(gpu_enable: bool, tensors: dict[int, torch.Tensor], args) -> pd.DataFrame:
  output_ids = args['output ids']
  result_names = args['output names']

  if gpu_enable:
    for output_id in output_ids:
      tensors[output_id].tensor = tensors[output_id].tensor.to('cpu', non_blocking=True)
    torch.cuda.synchronize()
  
  result = pd.DataFrame()

  tensor, type, _  = None, None, None

  for output_id, name in zip(output_ids, result_names):
    tensor, type = tensors[output_id]
    
    if type == 'int':
      result[name] = tensor.squeeze() if tensor.dim() != 1 else tensor

    elif type == 'float':
      result[name] = tensor.squeeze() if tensor.dim() != 1 else tensor
    
    elif type == 'string':
      if tensor.dim() == 1:
        result[name] = list(num_to_str(tensor))
      else:
        result[name] = [num_to_str(row) for row in tensor]
    
    elif type == 'date':
      result[name] = [float_to_date(row.item()) for row in tensor]

    else:
      raise ValueError(f"Unsupported tensor type: {type}")

  del tensor, type, _
  return result