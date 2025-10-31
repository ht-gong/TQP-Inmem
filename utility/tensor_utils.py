import torch
from typing import List
import gc
from utility.logger import message_logger


def find_boundaries_for_join(key_col_left_view, key_col_right_view, max_numel_per_chunk):
    assert key_col_left_view[0].tensor.dtype == key_col_right_view[0].tensor.dtype, 'Key columns must be of same type'
    # assert key_col_left_view[0].tensor[0] <= key_col_left_view[0].tensor[-1], 'Left key column is not ascending'
    # assert key_col_right_view[0].tensor[0] <= key_col_right_view[0].tensor[-1], 'Right key column is not ascending'
    assert key_col_left_view[0].tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8), "only join on ints supported"

    message_logger().critical("-----TENSOR INFO:--------")
    for k in key_col_left_view:
        message_logger().critical(k.state)
    for k in key_col_right_view:
        message_logger().critical(k.state)

    Ls = [k.tensor for k in key_col_left_view]
    Rs = [k.tensor for k in key_col_right_view]
    allT = Ls + Rs

    gpu_tensors = [t for t in allT if t.is_cuda]
    cpu_tensors = [t for t in allT if not t.is_cuda]

    N = sum(t.numel() for t in allT)
    if N == 0:
        return [[]], [[]]

    # number of interior cuts
    K = (N + max_numel_per_chunk - 1) // max_numel_per_chunk 
    if K <= 0:
        return [[t.numel()] for t in Ls], [[t.numel()] for t in Rs]  # one chunk

    # Targets: cumulative counts we want at each cut
    targets_cpu = torch.arange(1, K + 1, device='cpu', dtype=torch.int64) * max_numel_per_chunk
    
    targets_cpu.clamp_(max=N)  # last may equal N

    l = min(int(t[0].cpu().item()) for t in allT)
    h = max(int(t[-1].cpu().item()) for t in allT)

    lo_cpu = torch.full((K,), l, device='cpu', dtype=torch.int64)
    hi_cpu = torch.full((K,), h, device='cpu', dtype=torch.int64)

    # Vectorized batched binary search
    # Each iteration: compute counts(mid) for ALL cuts in parallel on GPU
    for _ in range(64):  # enough for 64-bit range; usually much fewer
        mid_cpu = lo_cpu + ((hi_cpu - lo_cpu) // 2)  # [K]
        if len(gpu_tensors) > 0:
            mid_gpu = mid_cpu.to(gpu_tensors[0].device)
        counts_cpu = torch.zeros_like(mid_cpu)

        for t in gpu_tensors:
            counts_cpu += torch.searchsorted(t, mid_gpu, right=True).to('cpu')  # [K] per tensor, summed
        for t in cpu_tensors:
            counts_cpu += torch.searchsorted(t, mid_cpu, right=True)

        gt = counts_cpu > targets_cpu
        # If counts < target → move lo up; else move hi down
        lo_cpu = torch.where(gt, lo_cpu, mid_cpu + 1)
        hi_cpu = torch.where(gt, mid_cpu, hi_cpu)

        if torch.equal(lo_cpu, hi_cpu):
            break

    cutvals_cpu = lo_cpu
    if len(gpu_tensors) > 0:
        cutvals_gpu = lo_cpu.to(gpu_tensors[0].device)

    cutoffs_left, cutoffs_right = [], []
    for t in Ls:
        if t.is_cuda:
            cutoffs_left.append(torch.searchsorted(t, cutvals_gpu, right=True).tolist())
        else:
            cutoffs_left.append(torch.searchsorted(t, cutvals_cpu, right=True).tolist())
    for t in Rs:
        if t.is_cuda:
            cutoffs_right.append(torch.searchsorted(t, cutvals_gpu, right=True).tolist())
        else:
            cutoffs_right.append(torch.searchsorted(t, cutvals_cpu, right=True).tolist())

    return cutoffs_left, cutoffs_right

def find_boundaries_for_merge(src_list, chunk_size):
    """
    Find merge boundaries (cutoff indices) for a collection of sorted tensors,
    supporting both ascending and descending order.

    Args:
        src_list (List[torch.Tensor]): List of 1D sorted tensors.
        chunk_size (int): The number of elements per merge chunk.
    
    Returns:
        List[List[int]]: A list (per tensor) of cutoff indices.
    """
    dtype = src_list[0].dtype
    N = sum(c.shape[0] for c in src_list)
    cutoffs = [[] for _ in range(len(src_list))]
    
    # Choose tolerance based on type. For floating point types, use 1e-5;
    # for integer types, a tolerance of 1 is sufficient.
    if dtype in (torch.float64, torch.float32):
        limit = 1e-5
    else:
        limit = 1

    # Determine the sorting order.
    # Here we assume that the entire src_list is sorted in the same order.
    is_ascending = src_list[0][0] <= src_list[0][-1]
    
    current_target = min(chunk_size // src_list[0].element_size(), N)
    processed_elems = 0
    iter_num = 0

    while True:
        # Set the initial binary search bounds based on the data order.
        if is_ascending:
            low_val = torch.iinfo(dtype).min if dtype in (torch.int64, torch.int8) else torch.tensor(-1e12, dtype=dtype)
            high_val = torch.iinfo(dtype).max if dtype in (torch.int64, torch.int8) else torch.tensor(1e12, dtype=dtype)
        else:
            # For descending, swap the roles so that low_val is the maximum value.
            low_val = torch.iinfo(dtype).max if dtype in (torch.int64, torch.int8) else torch.tensor(1e12, dtype=dtype)
            high_val = torch.iinfo(dtype).min if dtype in (torch.int64, torch.int8) else torch.tensor(-1e12, dtype=dtype)
        
        # Uncomment the line below to debug the initial bounds.
        # print("Initial bounds:", low_val, high_val)
        
        # Binary search to find the value which gives the desired overall count.
        while abs(high_val - low_val) > limit:
            mid = torch.tensor(torch.div(low_val + high_val, 2)).to(dtype)
            count = 0
            for arr in src_list:
                if is_ascending:
                    c = torch.searchsorted(arr, mid, right=True).item()
                else:
                    c = torch.searchsorted(-arr, -mid, right=False).item()
                count += c
            if count == current_target:
                low_val = mid
                break
            else:
                #if is_ascending:
                    # In ascending order, a lower count means we need to increase mid.
                    if count < current_target:
                        low_val = mid
                    else:
                        high_val = mid

        # Determine cutoff indices for each tensor.
        for i, arr in enumerate(src_list):
            if is_ascending:
                idx = torch.searchsorted(arr, torch.tensor(low_val, dtype=dtype), right=True).item()
            else:
                idx = torch.searchsorted(-arr, -torch.tensor(low_val, dtype=dtype), right=False).item()
            cutoffs[i].append(idx)

        # Compute how many elements have been processed (using the latest cutoff).
        processed_elems = sum(co[-1] for co in cutoffs)
        if processed_elems == N:
            break
        # Update the target for the next chunk.
        current_target = min(processed_elems + chunk_size, N)

    return cutoffs


def resize_tensor_list(subject:List[torch.tensor], target:List[torch.tensor]):
    if len(subject) != len(target):
        raise ValueError("Resize target and subject length mismatch")
    for i in range(len(target)):
        if subject[i].shape != target[i].shape:
            subject[i].resize_(target[i].shape)

def allocate_tensor_list(subject: List[torch.tensor], target: List[torch.tensor], mem_pool):
    if len(subject) != len(target):
        raise ValueError("Resize target and subject length mismatch")
    ptrs = []
    for i in range(len(target)):
        ptr, subject[i] = mem_pool.malloc_like(target[i])
        ptrs.append(ptr)
    return ptrs

def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def clean_tensors():
    # Run Python garbage collector
    gc.collect()

    # Empty PyTorch CUDA cache
    torch.cuda.empty_cache()

def show_tensor_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    print(f"Type: {type(obj)}, Size: {obj.size()}, Memory: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
        except:
            pass
    # print(f"Allocated: {torch.cuda.memory_allocated(torch.device('cuda:0')) / 1024**2:.2f} MB")
    # print(f"Reserved:  {torch.cuda.memory_reserved(torch.device('cuda:0')) / 1024**2:.2f} MB")

    print("===========================")

def tensor_lists_equal(list1, list2, device=None):
    """
    Compare two lists of tensors for equality after moving them to the same device.
    
    Args:
        list1 (List[torch.Tensor]): First list of tensors.
        list2 (List[torch.Tensor]): Second list of tensors.
        device (torch.device or str, optional): If specified, move all tensors to this device.
                                                If None, use the device of list1[0].

    Returns:
        bool: True if all corresponding tensors are equal, False otherwise.
    """
    if len(list1) != len(list2):
        return False

    if device is None:
        device = list1[0].device

    for t1, t2 in zip(list1, list2):
        t1 = t1.to(device)
        t2 = t2.to(device)
        if not torch.equal(t1, t2):
            return False

    return True