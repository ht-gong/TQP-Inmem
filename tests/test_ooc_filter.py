import sys
import os
import time
import torch
import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from operators.filter import filter_vortex
from variable import Variable

if __name__ == "__main__":
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')

    rows = 10_0000_0000

    # print(f"Memory allocated: {torch.cuda.memory_allocated() / 2**30:.4f} GiB")
    tensorsa = {
      1: Variable(torch.arange(rows), ''),
      2: Variable(torch.arange(rows), '')
    }
    low, up = 10000, 20000
    args = {}
    args['input ids'] = [1, 2]
    args['filter tree'] = [['b#1', '>=', f'{low}'], 'AND', ['a#2', '<=', f'{up}']]

    print ("Start!")

    start = time.time()

    tensorsb = {}
    res_rows = filter_vortex(True, tensorsb, tensorsa, args, None)

    end = time.time()
    # print (f"skewness = {skewness}, join time = {end - start:.4f} s, {rows} rows")
    print (f"time = {end - start:.4f} s, {res_rows} rows")
    if res_rows != up - low + 1:
        print (f"WRONG: {res_rows} != {up - low + 1}")

    print (tensorsb[1].tensor[0:20])
    print (tensorsb[2].tensor[0:20])
    # print (tensorsa[1].tensor[:])
    # print (tensorsb[2].tensor[:])