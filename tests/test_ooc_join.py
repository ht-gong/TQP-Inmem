import sys
import os
import time
import torch
import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from operators.hashjoin import join_vortex
from variable import Variable
# from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

import numpy as np

def generate_zipfian_data(num_rows, num_keys, skew_factor=1.0001):
    zipfian_keys = np.random.zipf(skew_factor, num_rows) % num_keys
    return torch.tensor(zipfian_keys, dtype=torch.int64)
def test_inner():
    rows = 10_0000_0000
    for dup in [4]:
        tensorsa = {
            1: Variable(torch.arange(rows), '')
        }
        tensorsb = {
            2: Variable(torch.arange(rows * dup) // dup, '')
        }
        print ("Start!")

        start = time.time()
        res_rows = join_vortex(True, tensorsa, tensorsb, 1, 2, 1 << 27, [], 'inner')

        end = time.time()
        print (f"dup = {dup}, join time = {end - start:.4f} s, {res_rows} rows")

        if rows * 4 != res_rows:
            print ("wrong out.")
        print (tensorsa[1].tensor[0:20])
        print (tensorsb[2].tensor[0:20])

def test_outer():
    tensorsb = {
        1: Variable(torch.tensor([1, 2, 5, 7, 9]), ''),
        11: Variable(torch.tensor([11, 22, 55, 77, 99]), '')
    }
    tensorsa = {
        2: Variable(torch.tensor([1, 3, 7, 8, 10]), ''),
        22: Variable(torch.tensor([6, 6, 6, 6, 6]), '')
    }
    print ("Start!")

    start = time.time()
    res_rows = join_vortex(True, tensorsa, tensorsb, 2, 1, 1 << 27, [], 'right-outer')

    end = time.time()
    print (f"join time = {end - start:.4f} s, {res_rows} rows")

    print (tensorsa[2].tensor[0:20])
    print (tensorsa[22].tensor[0:20])
    print (tensorsb[1].tensor[0:20])
    print (tensorsb[11].tensor[0:20])

def test_semi_anti():
    tensorsb = {
        1: Variable(torch.tensor([1, 2, 5, 7, 9]), ''),
        11: Variable(torch.tensor([11, 22, 55, 77, 99]), '')
    }
    tensorsa = {
        2: Variable(torch.tensor([1, 1, 7, 8, 10]), ''),
        22: Variable(torch.tensor([6, 6, 6, 6, 6]), '')
    }
    print ("Start!")

    start = time.time()
    res_rows = join_vortex(True, tensorsa, tensorsb, 2, 1, 1 << 27, [], 'right-anti') # 'right-semi'

    end = time.time()
    print (f"join time = {end - start:.4f} s, {res_rows} rows")

    print (tensorsb[1].tensor[0:20])
    print (tensorsb[11].tensor[0:20])

if __name__ == "__main__":
    torch.cuda.set_device(0)
    # test_outer()
    test_semi_anti()
    