import pandas as pd
import numpy as np
import re
import torch

from torch import Tensor
from itertools import accumulate
import constants

YEAR_START = 1990
YEAR_END = 2000

epoch = pd.Timestamp(f'{YEAR_START}-01-01')
year_range = list(range(YEAR_START, YEAR_END))
year_edges = list(accumulate([365 if year % 4 else 366 for year in year_range]))

# num_of_ns_per_year = 365.2 * 24 * 60 * 60 * (10**9)

def is_date(s: str):
  return len(s) == 10 and s.count('-') == 2 and s.replace('-', '').isdigit()

def is_float(s: str):
  return s.replace('.', '').replace('-', '').replace('E', '').isdigit()

def date_to_float(date_str: str):
  date = pd.Timestamp(date_str)
  diff = (date - epoch).days
  return diff

def float_to_date(diff: float):
  return (epoch + pd.Timedelta(days=diff)).strftime('%Y-%m-%d')

def float_to_year(diff: float):
  return int((epoch + pd.Timedelta(days=diff)).year)

def str_to_np(s, n):
  return np.pad([ord(c) for c in s], (0, n - len(s)), constant_values=0)

def num_to_str(tensor: Tensor):
  s = [chr(ele) if ele != 0 else '~' for ele in tensor]
  return "".join(s).rstrip('~')

def append_nulls(device_name, tensors, num):
  if num <= 0:
    assert num == 0
    return 
  
  # print (f"appending {num} null values")
  for v in tensors.keys():
    tensors[v].tensor = torch.cat((tensors[v].tensor, torch.full((num, ), constants.null, dtype=tensors[v].tensor.dtype, device=device_name)))

def index_with_null(data, idx):
  mask = idx == constants.null 
  safe_idx = idx.masked_fill(mask, 0)          # 0 just needs to be a valid index
  data = data[safe_idx]
  data.masked_fill_(mask, constants.null) 
  return data 

def rearrange_tensors(tensors, outidx):
  for v in tensors.keys():
    tensors[v].tensor = tensors[v].tensor[outidx]

def test():
  # EXTRACT (YEAR) Approximation
  for days in range(365, 365 * 10):
    y = (epoch + pd.Timedelta(days=days)).year
    if y >= 1999:
      break
    if y < 1992:
      continue
    # if y != math.floor(1990 + (days / o)):
    #   print (days, 1990 + (days / o), (epoch + pd.Timedelta(days=days)).strftime('%Y-%m-%d'))
    bin_idx = torch.bucketize(torch.tensor(days), torch.tensor(year_edges), right=True)
    res = torch.tensor(year_range)[bin_idx]
    if y != res:
      print (days, (epoch + pd.Timedelta(days=days)).strftime('%Y-%m-%d'))