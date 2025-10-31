import torch
import constants
from enum import Enum

class VariableState(str, Enum):
  CPU = 'cpu resident'
  GPU = 'gpu resident'

class Variable:
  def __init__(self, tensor: torch.Tensor, tensor_type: str, backing_mem_ptr = None, var_state: VariableState=VariableState.CPU, is_sorted: bool = False):
    self.tensor = tensor
    self.tensor_type = tensor_type
    self.state = var_state
    self.backing_mem_ptr = backing_mem_ptr
    self.is_sorted = is_sorted

  def __repr__(self):
    return f"Variable (tensor size = {self.tensor.shape} ({self.tensor.dtype}), tensor type = '{self.tensor_type}')"
  
  def __iter__(self):
    yield self.tensor
    yield self.tensor_type

  def __getitem__(self, index):
    sliced = self.tensor[index]
    return Variable(sliced, self.tensor_type, self.backing_mem_ptr)

  def free_underlying_mem(self, cpu_pool, gpu_pool):
    if self.backing_mem_ptr:
      if self.state == VariableState.GPU:
        gpu_pool.free(self.backing_mem_ptr)
      else:
        cpu_pool.free(self.backing_mem_ptr)

  def normalize(self):
    if self.tensor.dim() == 2:
      self.tensor = self.tensor.squeeze()
    if self.tensor.dim() == 0:
      self.tensor = self.tensor.unsqueeze(0)
    assert self.tensor.dim() == 1

  def get_type_torch(self):
    return [type_to_torch(self.tensor_type)]

def torch_to_type(dtype):
  if dtype == constants.float_dtype:
    return 'float'
  if dtype == constants.date_dtype:
    return 'date'
  if dtype == constants.int_dtype:
    return 'int'
  return 'string'

def type_to_torch(dtype):
  if dtype == 'float':
    return constants.float_dtype
  if dtype == 'int':
    return constants.int_dtype
  if dtype == 'date':
    return constants.date_dtype
  return constants.string_dtype
