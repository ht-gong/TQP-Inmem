from calendar import c
import time
import torch
import pandas as pd

from conversion import date_to_float, is_date, is_float, str_to_np 
from operators.like import like_contains, like_mask
from variable import Variable
import constants
from IO.vortex_pipeline import InMemoryPipeline 
from parsing import get_literals, get_id
from utility.logger import perf_logger, message_logger

def evaluate(device_name, tensors, tree, rows, subquery_result, leaf_function=None):
  if rows == None:
    rows = list(tensors.values())[0].tensor.size(0)

  if isinstance(tree, str): # date, number, or str
    if tree == 'Subquery':
      return subquery_result

    if is_date(tree):
      return date_to_float(tree)
    
    if is_float(tree):  # TODO: what if the string is a number?
      return float(tree) if '.' in tree else int(tree)
    
    if '#' in tree and not tree.startswith('Brand#'):
      # e.g., p_brand#30 = Brand#12
      if leaf_function:
        return leaf_function(get_id(tree))
      
      return tensors[get_id(tree)].tensor.squeeze()

    return tree

  if len(tree) == 1:    # Redundant Brackets
    return evaluate(device_name, tensors, tree[0], rows, subquery_result, leaf_function)
   
  if len(tree) == 3 and tree[1] == '#':
    # e.g., [['0.2', '*', ['avg', 'l_quantity']], '#', '30']]
    if leaf_function:
      return leaf_function(int(tree[-1]))
    return tensors[int(tree[-1])].tensor.squeeze()

  if len(tree) == 2:
    if tree[0] == 'StartsWith':
      # ['StartsWith', ['p_type#31', 'MEDIUM POLISHED']]
      id, name = get_id(tree[1][0]), tree[1][1]
      t = tensors[id].tensor
      assert t.shape[-1] >= len(name)
      p = torch.tensor(str_to_np(name, len(name)), dtype=constants.string_dtype, device=device_name)
      o = torch.eq(t[:, :len(name)], p).all(dim=1)
      # message_logger().debug(f"{o.shape} satisfy {o.sum()} / {rows}")
      return o
    if tree[0] == 'EndsWith' or tree[0] == 'Contains':
      # EndsWith: Substring instead ...
      id, name = get_id(tree[1][0]), tree[1][1]
      return like_contains(
        tensors[id].tensor,
        torch.tensor(str_to_np(name, len(name)), device=device_name)
      )
    
    if tree[0] == 'NOT':
      return ~ evaluate(device_name, tensors, tree[1], rows, subquery_result, leaf_function)
    
    if tree[0] == 'isnotnull':
      message_logger().debug("SUPPOSING NO NULL VALUES.")
      return torch.ones(rows, dtype=torch.bool, device=device_name)
    
    if tree[0] == 'substring':
      # e.g., ['substring', ['c_phone#317', '1', '2']]
      assert len(tree[1]) == 3
      sub, l, w = tree[1]
      l, w = int(l) - 1, int(w)
      return evaluate(device_name, tensors, sub, rows, subquery_result, leaf_function)[:, l:l+w]
    
    if tree[0] == 'cast':
      # 'ps_availqty#97 as double'
      literal, _, cast_type = tree[1][0].split(' ')
      if cast_type == 'double':
        # TODO: double it
        return evaluate(device_name, tensors, literal, rows, subquery_result, leaf_function)
        # return evaluate(device_name, tensors, literal, rows, subquery_result, leaf_function).double()
      
    raise ValueError(f"Unsupported Function: {tree[0]}")
  
  op = tree[-2]

  if op == 'IN': # IN Not in Aggregation
    if tree == ['DELIVER', 'IN', 'PERSON']: # 😅
      return " ".join(tree)
    # Suppose no mix of single and multiple characters
    # e.g., ['p_container#33', 'IN', ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']]
    L = evaluate(device_name, tensors, tree[0], rows, subquery_result, leaf_function)
    if L.dtype == constants.string_dtype:
      R = [name for name in tree[-1]]
    elif L.dtype in (constants.int_dtype, constants.date_dtype):
      R = [int(name) for name in tree[-1]]
    else:
      assert L.dtype == constants.float_dtype
      R = [float(name) for name in tree[-1]]

    if L.dim() == 1:
      L.unsqueeze_(0)

    mask = torch.zeros(rows, dtype=torch.bool, device=device_name)

    for r in R:
      if isinstance(r, str):
        if len(r) == 1:
          assert L.squeeze().dim() != 2
          mask = torch.logical_or(mask, torch.eq(L, ord(R)))
        else:
          t = torch.tensor(str_to_np(r, L.shape[-1]), dtype=constants.string_dtype, device=device_name)
          mask = torch.logical_or(mask, (L == t).all(dim=1))
      else:
        mask = torch.logical_or(mask, torch.eq(L, r))
    return mask

  if op == 'LIKE':
    L = evaluate(device_name, tensors, tree[0], rows, subquery_result, leaf_function)
    assert tree[-1].count('%') == 3
    _, word1, word2, _ = tree[-1].split('%')
    word1 = torch.tensor(str_to_np(word1, len(word1)), dtype=constants.string_dtype, device=device_name)
    word2 = torch.tensor(str_to_np(word2, len(word2)), dtype=constants.string_dtype, device=device_name)
    return like_mask(L, word1, word2)

  ops = {
    '<=': torch.le,
    '>=': torch.ge,
    '<': torch.lt,
    '>': torch.gt,
    '=': torch.eq,
    'AND': torch.logical_and,
    'OR': torch.logical_or
  }

  if op not in ops:
    raise ValueError(f"Unsupported Operation: {op}")

  L = evaluate(device_name, tensors, tree[0], rows, subquery_result, leaf_function)
  R = evaluate(device_name, tensors, tree[-1], rows, subquery_result, leaf_function)

  if isinstance(R, float):
    if op == '<=':
      return torch.le(L, R + constants.eps)
    if op == '>=':
      return torch.ge(L, R - constants.eps)
  
  if isinstance(R, str):    # DO NOT SUPPORT STRING < and >
    if len(R) == 1:
      return ops[op](L, ord(R))
    else:
      if L.dim() == 1:
        L.unsqueeze_(0)
      R = torch.tensor(str_to_np(R, L.shape[-1]), dtype=constants.string_dtype, device=device_name)
      return (L == R).all(dim=1)

  if op == '=' and L.dtype == constants.float_dtype:
    return torch.isclose(L, torch.full((L.shape[0],), R, dtype=L.dtype, device=device_name) if isinstance(R, float) else R, atol=constants.eps)
  return ops[op](L, R)

def tqp_filter(gpu_enable, tensor_group, args, subquery_result, cpu_mem_pool, gpu_mem_pool, name):
  perf_logger().start(name)

  assert 'mask map' not in args, "Filter not accepting masks"

  device_name = 'cuda' if gpu_enable else 'cpu'
  tree = args['filter tree']
  input_columns = {k: tensor_group[k] for k in args['materialized in']}

  if len(input_columns) == 0:
    message_logger().debug("Filter not dispatched")
    perf_logger().stop(name)
    return -1

  def filter_wrapper(cols):
    mask = evaluate(device_name, cols, tree, None, subquery_result)
    return (mask,)
  
  pipe = InMemoryPipeline(
    input_columns=input_columns,
    output_columns_type=[torch.bool],
    operator=filter_wrapper,
    chunk_size=2_000_000_000,
    cpu_mem_pool=cpu_mem_pool,
    gpu_mem_pool=gpu_mem_pool,
    name=f"{name} filter pipe"
  )
  pipe.do_exchange(20_000_000)

  mask = pipe.get_result()[0]

  def squeeze(tensor):
    if tensor.dim() == 1:
      return tensor
    if tensor.dim() == 2 and tensor.shape[0] == 1:
      return tensor.squeeze(0)
    assert False, "squeeze failed."

    # Store mask in mask id
  tensor_group[args['out mask id']] = Variable(squeeze(mask.tensor), 'bool', mask.memory_ptr, mask.placement)
  perf_logger().stop(name)
  return -1