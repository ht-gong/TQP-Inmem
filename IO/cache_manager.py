import torch 
from utility.logger import message_logger

def zero_copy_mask_policy(mask_col, data_col):
    assert mask_col.dtype == torch.bool, "Mask Column bust be boolean tyoe"
    other_dims = 1
    for i in data_col.shape[1:]:
        other_dims *= i
    apply = data_col.is_cpu and mask_col.is_cuda and mask_col.sum() / mask_col.numel()  < min(0.2 * other_dims, 0.5)
    if apply:
        message_logger().critical("ZERO COPY MASK applied")
    return apply

def zero_copy_rearrange_policy(rearrange, data_col):
    assert rearrange.dtype == torch.int64, "Rearrange Column bust be torch.int64 tyoe"
    other_dims = 1
    for i in data_col.shape[1:]:
        other_dims *= i
    apply = data_col.is_cpu and rearrange.is_cuda and rearrange.numel() / data_col.shape[0] < min(0.2 * other_dims, 0.5)
    if apply:
        message_logger().critical("ZERO COPY REARR applied")
    # return False
    return apply

def generate_cache_hints(query_plan_parent_info, query_plan_operator_info):
    pass