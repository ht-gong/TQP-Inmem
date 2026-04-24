import torch
from variable import Variable
from utility.logger import perf_logger


def tqp_sort(gpu_enable, tensor_group, args, with_limit, name="sort"):
  perf_logger().start(name)
  device_name = 'cuda' if gpu_enable else 'cpu'

  if with_limit:
    limit = args['limit']

  input_ids = args['input ids']
  rows = tensor_group[input_ids[0]].tensor.shape[0]

  key_ids, key_descs = args['key ids'], args['key orders']

  if tensor_group[key_ids[0]].tensor.numel() * \
     tensor_group[key_ids[0]].tensor.element_size() < 8_000_000_000:
    perm = torch.arange(rows, device=device_name).contiguous()

    for key_id, desc in zip(reversed(key_ids), reversed(key_descs)):
      key_col = tensor_group[key_id].tensor.contiguous()
      key_col_d = torch.empty_like(key_col, device=device_name).contiguous()
      key_col_d.copy_(key_col)

      if key_col.dim() == 2:
        tmp = torch.gather(key_col_d, 0, perm.unsqueeze(1).expand_as(key_col_d))
        uni, inv = torch.unique(tmp, sorted=True, return_inverse=True, dim=0)
        sorted_tmp, inverse_index = torch.sort(inv, stable=True, descending=desc)
        perm = torch.gather(perm, 0, inverse_index)
      else:
        tmp = torch.gather(key_col_d, 0, perm)
        sorted_tmp, inverse_index = torch.sort(tmp, stable=True, descending=desc)
        perm = torch.gather(perm, 0, inverse_index)

    torch.cuda.synchronize()
  else:
    raise RuntimeError("sort input exceeds in-memory limit")

  if with_limit:
    perm = perm[:limit]

  for key_id in input_ids:
    if tensor_group[key_id].tensor.dim() == 2:
      tmp = perm.unsqueeze(1).expand(-1, tensor_group[key_id].tensor.shape[-1])
    else:
      tmp = perm

    if tensor_group[key_id].tensor.device != tmp:
      tensor_group[key_id].tensor = tensor_group[key_id].tensor.to(tmp.device)

    tensor_group[key_id] = Variable(torch.gather(tensor_group[key_id].tensor, 0, tmp),
                                    tensor_group[key_id].tensor_type)

  del tmp, sorted_tmp, inverse_index
  perf_logger().stop(name)
  return rows
