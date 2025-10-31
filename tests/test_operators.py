import sys
import os
import torch
import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from operators.hashjoin import tqp_hashjoin
from variable import Variable
from operators.sortjoin import tqp_sortjoin, tqp_sortjoin_outer, tqp_sortjoin_semi_or_anti

gpu_enable = True

### =========== SORT ============ ###




### =========== JOIN ============ ###

@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb",
    [
        (
            {1: torch.tensor([0, 0, 1, 1])},
            {2: torch.tensor([0, 1, 2, 2])},
            1,
            2,
            {1: torch.tensor([0, 0, 1, 1])},  # Expected values after sortjoin (replace with correct expected)
            {2: torch.tensor([0, 0, 1, 1])},
        ),
        (
            {1: torch.tensor([1, 1, 3, 5, 4]), 3: torch.tensor([2, 4, 6, 8, 10])},
            {2: torch.tensor([5, 3, 1, 9]), 4: torch.tensor([1, 3, 5, 7])},
            1,
            2,
            {1: torch.tensor([1, 1, 3, 5]), 3: torch.tensor([2, 4, 6, 8])},
            {2: torch.tensor([1, 1, 3, 5]), 4: torch.tensor([5, 5, 3, 1])},
        ),
    ],
)
# @pytest.mark.skip(reason="Temporarily disabled")
def test_tqp_sortjoin(tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb):
    # Move tensors to CUDA
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')
    
        torch.cuda.synchronize()

    # Apply the function
    tqp_sortjoin(gpu_enable, ta, tb, key_a, key_b)

    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"
    for key in expected_tensorsb.keys():
        assert torch.equal(tb[key].tensor.cpu(), expected_tensorsb[key]), f"Mismatch in tensorsb[{key}]"


### =========== SEMI ============ ###

@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa",
    [
        (
            {1: torch.tensor([1, 2, 1, 2]), 3: torch.tensor([4, 7, 9, 10])},
            {2: torch.tensor([0, 1, 3, 9, 1]), 4: torch.tensor([2, 2, 2, 3, 1])},
            1,
            2,
            {1: torch.tensor([1, 1]), 3: torch.tensor([4, 9])},  # Expected values after sortjoin (replace with correct expected)
        ),
    ],
)
def test_tqp_leftsemi(tensorsa, tensorsb, key_a, key_b, expected_tensorsa):
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')

        torch.cuda.synchronize()

    # Apply the function
    tqp_sortjoin_semi_or_anti(gpu_enable, ta, tb, key_a, key_b, 'semi')

    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"

### =========== ANTI ============ ###

@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa",
    [
        # Test Case 1: Some values in tensorsa[key_a] do not exist in tensorsb[key_b] with duplicates
        (
            {1: torch.tensor([1, 2, 3, 4, 1, 2, 3, 4]), 3: torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])},
            {2: torch.tensor([2, 4, 6, 4, 2]), 4: torch.tensor([5, 15, 25, 35, 45])},
            1,
            2,
            {1: torch.tensor([1, 1, 3, 3]), 3: torch.tensor([10, 50, 30, 70])},  # Only 1 and 3 remain
        ),

        # Test Case 2: All values in tensorsa[key_a] exist in tensorsb[key_b], with multiple repetitions
        (
            {1: torch.tensor([5, 6, 5, 6, 7, 7]), 3: torch.tensor([50, 60, 50, 60, 70, 70])},
            {2: torch.tensor([5, 6, 7, 5, 6, 7]), 4: torch.tensor([100, 200, 300, 400, 500, 600])},
            1,
            2,
            {1: torch.tensor([]), 3: torch.tensor([])},  # No values left
        ),

        # Test Case 3: No values in tensorsa[key_a] exist in tensorsb[key_b] (should return original tensorsa)
        (
            {1: torch.tensor([8, 9, 10]), 3: torch.tensor([80, 90, 100])},
            {2: torch.tensor([1, 2, 3]), 4: torch.tensor([10, 20, 30])},
            1,
            2,
            {1: torch.tensor([8, 9, 10]), 3: torch.tensor([80, 90, 100])},  # No matches, return unchanged
        ),

        # Test Case 4: Some values in tensorsa[key_a] exist multiple times in tensorsb[key_b], should remove all occurrences
        (
            {1: torch.tensor([1, 2, 2, 3, 4, 2]), 3: torch.tensor([11, 22, 33, 44, 55, 66])},
            {2: torch.tensor([2, 4, 2]), 4: torch.tensor([100, 200, 300])},
            1,
            2,
            {1: torch.tensor([1, 3]), 3: torch.tensor([11, 44])},  # 2 and 4 removed completely
        ),
    ],
)
def test_tqp_leftanti(tensorsa, tensorsb, key_a, key_b, expected_tensorsa):
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')

        torch.cuda.synchronize()

    # Apply the function
    tqp_sortjoin_semi_or_anti(gpu_enable, ta, tb, key_a, key_b, 'anti')

    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"


### =========== OUTER ============ ###

@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb",
    [
        # Test Case 1: Some values in tensorsa[key_a] have matches, others do not
        (
            {1: torch.tensor([1, 2, 3, 4]), 3: torch.tensor([10, 20, 30, 40])},
            {2: torch.tensor([2, 4]), 4: torch.tensor([200, 400])},
            1,
            2,
            {1: torch.tensor([2, 4, 1, 3]), 3: torch.tensor([20, 40, 10, 30])},  # Sorted non-null, then null
            {2: torch.tensor([2, 4, constants.null, constants.null]), 4: torch.tensor([200, 400, constants.null, constants.null])},
        ),

        # Test Case 2: All values in tensorsa[key_a] exist in tensorsb[key_b] (full join)
        (
            {1: torch.tensor([5, 6, 7]), 3: torch.tensor([50, 60, 70])},
            {2: torch.tensor([5, 6, 7]), 4: torch.tensor([500, 600, 700])},
            1,
            2,
            {1: torch.tensor([5, 6, 7]), 3: torch.tensor([50, 60, 70])},
            {2: torch.tensor([5, 6, 7]), 4: torch.tensor([500, 600, 700])},
        ),

        # Test Case 3: No values in tensorsa[key_a] exist in tensorsb[key_b] (all right columns should be constants.null at the end)
        (
            {1: torch.tensor([8, 9, 10]), 3: torch.tensor([80, 90, 100])},
            {2: torch.tensor([1, 2, 3]), 4: torch.tensor([10, 20, 30])},
            1,
            2,
            {1: torch.tensor([8, 9, 10]), 3: torch.tensor([80, 90, 100])},
            {2: torch.tensor([constants.null, constants.null, constants.null]), 4: torch.tensor([constants.null, constants.null, constants.null])},  # All nulls at the end
        ),

        # Test Case 4: Some values in tensorsa[key_a] appear multiple times and have multiple matches
        (
            {1: torch.tensor([1, 2, 2, 3]), 3: torch.tensor([11, 22, 33, 44])},
            {2: torch.tensor([2, 3]), 4: torch.tensor([200, 300])},
            1,
            2,
            {1: torch.tensor([2, 2, 3, 1]), 3: torch.tensor([22, 33, 44, 11])},
            {2: torch.tensor([2, 2, 3, constants.null]), 4: torch.tensor([200, 200, 300, constants.null])},  # Null (constants.null) at the end
        ),

        # Test Case 5: All values in tensorsa[key_a] have multiple matches in tensorsb[key_b] (should duplicate rows)
        (
            {1: torch.tensor([1, 2]), 3: torch.tensor([10, 20])},
            {2: torch.tensor([2, 2, 2]), 4: torch.tensor([200, 201, 202])},
            1,
            2,
            {1: torch.tensor([2, 2, 2, 1]), 3: torch.tensor([20, 20, 20, 10])},
            {2: torch.tensor([2, 2, 2, constants.null]), 4: torch.tensor([200, 201, 202, constants.null])},  # Null (constants.null) last
        ),
    ],
)
def test_tqp_outer(tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb):
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')

        torch.cuda.synchronize()

    # Apply the function
    tqp_sortjoin_outer(gpu_enable, ta, tb, key_a, key_b)

    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"
    for key in expected_tensorsb.keys():
        assert torch.equal(tb[key].tensor.cpu(), expected_tensorsb[key]), f"Mismatch in tensorsb[{key}]"

def reorder_tensors(tensors, key):
    sorted_tensor, idx = torch.sort(tensors[key].tensor)
    for col in tensors.keys():
        tensors[col].tensor = tensors[col].tensor[idx]


@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb",
    [
        (
            {1: torch.tensor([0, 0, 1, 1])},
            {2: torch.tensor([0, 1, 2, 2])},
            1,
            2,
            {1: torch.tensor([0, 0, 1, 1])},  # Expected results must be sorted 
            {2: torch.tensor([0, 0, 1, 1])},
        ),
        (
            {1: torch.tensor([1, 1, 3, 5, 4]), 3: torch.tensor([2, 4, 6, 8, 10])},
            {2: torch.tensor([5, 3, 1, 9]), 4: torch.tensor([1, 3, 5, 7])},
            1,
            2,
            {1: torch.tensor([1, 1, 3, 5]), 3: torch.tensor([2, 4, 6, 8])},
            {2: torch.tensor([1, 1, 3, 5]), 4: torch.tensor([5, 5, 3, 1])},
        ),
    ],
)
def test_tqp_hashjoin(tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb):
    # Move tensors to CUDA
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')
    
        torch.cuda.synchronize()

    # Apply the function
    tqp_hashjoin(gpu_enable, ta, tb, key_a, key_b, 3)

    reorder_tensors(ta, key_a)
    reorder_tensors(tb, key_b)
    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"
    for key in expected_tensorsb.keys():
        assert torch.equal(tb[key].tensor.cpu(), expected_tensorsb[key]), f"Mismatch in tensorsb[{key}]"

@pytest.mark.parametrize(
    "tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb, equileft, equiright",
    [
        (
            {1: torch.tensor([1, 1, 3, 5, 4]), 3: torch.tensor([2, 4, 5, 8, 10])},
            {2: torch.tensor([5, 3, 1, 9]), 4: torch.tensor([1, 3, 5, 7])},
            1,
            2,
            {1: torch.tensor([]), 3: torch.tensor([])},
            {2: torch.tensor([]), 4: torch.tensor([])},
            3,
            4,
        ),
        (
            {1: torch.tensor([1, 2, 3, 4, 5, 6, 7]), 3: torch.tensor([10, 20, 30, 40, 50, 60, 70])},
            {2: torch.tensor([9, 3, 5, 7, 8]), 4: torch.tensor([90, 30, 50, 70, 80])},
            1,
            2,
            {1: torch.tensor([3, 5, 7]), 3: torch.tensor([30, 50, 70])},
            {2: torch.tensor([3, 5, 7]), 4: torch.tensor([30, 50, 70])},
            3,
            4,
        ),
        (
            {1: torch.tensor([1, 2, 3, 3, 5, 6, 7]), 3: torch.tensor([10, 20, 30, 30, 50, 60, 70])},
            {2: torch.tensor([8, 5, 3 ,3 ,3]), 4: torch.tensor([80, 50, 30, 30, 40])},
            1,
            2,
            {1: torch.tensor([3, 3, 3, 3, 5]), 3: torch.tensor([30, 30, 30, 30, 50])},
            {2: torch.tensor([3, 3, 3, 3, 5]), 4: torch.tensor([30, 30, 30, 30, 50])},
            3,
            4,
        )
    ],
)
def test_tqp_hashjoin(tensorsa, tensorsb, key_a, key_b, expected_tensorsa, expected_tensorsb, equileft, equiright):
    # Move tensors to CUDA
    ta, tb = {}, {}

    if not gpu_enable:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col], '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col], '')

    else:
        for col in tensorsa.keys():
            ta[col] = Variable(tensorsa[col].cuda(non_blocking=True), '')
        for col in tensorsb.keys():
            tb[col] = Variable(tensorsb[col].cuda(non_blocking=True), '')
    
        torch.cuda.synchronize()

    # Apply the function
    tqp_hashjoin(gpu_enable, ta, tb, key_a, key_b, 3, equileft, equiright)

    reorder_tensors(ta, key_a)
    reorder_tensors(tb, key_b)
    # Verify results
    for key in expected_tensorsa.keys():
        assert torch.equal(ta[key].tensor.cpu(), expected_tensorsa[key]), f"Mismatch in tensorsa[{key}]"
    for key in expected_tensorsb.keys():
        assert torch.equal(tb[key].tensor.cpu(), expected_tensorsb[key]), f"Mismatch in tensorsb[{key}]"
