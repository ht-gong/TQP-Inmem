from typing import List, Literal
from torch import Tensor, device

from IO.vortex_pipeline import InMemoryPipeline
from conversion import index_with_null
from operators.filter import evaluate
from utility.logger import perf_logger, message_logger
from variable import Variable
import torch
import constants

LEFT = 0
RIGHT = 1

def evaluate_join(device_name, tensorsa, tensorsb, tree, leftidx, rightidx):
    def leaf(id: int):
        if id in tensorsa.keys():
            return tensorsa[id].tensor[leftidx].squeeze()
        else:
            return tensorsb[id].tensor[rightidx].squeeze()
        
    return evaluate(device_name, None, tree, leftidx.size(0), None, leaf)
  
def join_kernel_new(gpu_enable, tensors_left, tensors_right, left_key, right_key, condition_list, join_output_keys,
                return_global_index, type: Literal['inner', 'right-semi', 'right-anti', 'right-outer'], 
                left_global_idx, right_global_idx):
    assert type in {'inner', 'right-semi', 'right-anti', 'right-outer'}, "Invalid Join type."
    device_name = 'cuda' if gpu_enable else 'cpu'
    device = torch.device(device_name)
  
    tensors_left[left_key].normalize()
    tensors_right[right_key].normalize()
        
    left = tensors_left[left_key].tensor
    right = tensors_right[right_key].tensor
    nL, nR = left.size(0), right.size(0)
    # message_logger().debug("left.size, right.size, %s, %s", nL, nR)

    # Build index on left (S)
    # sort keys and ids
    left_keys = left
    left_ids = torch.arange(nL, device=device, dtype=torch.int64)
    sorted_keys, perm = torch.sort(left_keys)
    sorted_ids = left_ids[perm]
    left_ids = None

    # unique_consecutive + counts -> unique index + bucket counts
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    sorted_keys, perm = None, None
    # pre_fix: start position of each bucket in sorted_ids
    pre_fix = torch.cumsum(counts, dim=0) - counts  # same length as unique_keys

    # Probe: find for each right key whether it exists in unique_keys
    # searchsorted gives insertion point; check equality to filter matches
    # Ensure unique_keys is 1D on device
    pos = torch.searchsorted(unique_keys, right)
    # pos can equal len(unique_keys) for greater-than-all; mask safe-check
    in_range = pos < unique_keys.size(0)
    eq_mask = torch.zeros(nR, dtype=torch.bool, device=device)
    eq_mask[in_range] = unique_keys[pos[in_range]] == right[in_range]
    unique_keys = None

    # matched right indices and their bucket ids
    matched_right_idx = torch.nonzero(eq_mask, as_tuple=False).view(-1)

    if matched_right_idx.numel() > 0:
        bucket_ids = pos[matched_right_idx]  # which unique bucket each matched right belongs to

        # For each matched right (r_i) we need start and len of its bucket in sorted_ids
        startR = pre_fix[bucket_ids]        # shape (m,)
        lenR   = counts[bucket_ids]         # shape (m,)

        # compute candidate pairs using loop-unrolling idea (broadcast arange + masked_select)
        pos, pre_fix, in_range, eq_mask = None, None, None, None
        lmax = int(lenR.max().item())
        increments = torch.arange(lmax, device=device, dtype=torch.int64)  # (lmax,)
        
        # idx_mat: (m, lmax) = startR.view(m,1) + increments
        idx_mat = startR.view(-1, 1) + increments.view(1, -1)
        mask = increments.view(1, -1) < lenR.view(-1, 1)   # valid positions in each row
        idx_flat = torch.masked_select(idx_mat.view(-1), mask.view(-1)).to(torch.int64)
        startR, increments, idx_mat, mask = None, None, None, None 
        # CS: candidate left ids (from sorted_ids)
        if idx_flat.numel() > 0:
            CS = sorted_ids[idx_flat]                   # left row ids repeated per candidate
            sorted_ids = None
        else:
            CS = torch.tensor([], dtype=torch.int64, device=device)

        # CR: repeated right ids according to lenR
        if lenR.numel() > 0:
            CR = torch.repeat_interleave(matched_right_idx, lenR)  # right row ids aligned with CS
            matched_right_idx = None
        else:
            CR = torch.tensor([], dtype=torch.int64, device=device)

        idx_flat = None
        # Apply join key equality first (cheap)
        left_keys_sel = left[CS]
        right_keys_sel = right[CR]
        match_mask = left_keys_sel == right_keys_sel
        left_keys_sel, right_keys_sel = None, None

        # Apply additional conditions (call your evaluate_join for each condition).
        # evaluate_join returns a boolean mask aligned with validLeftIdx / validRightIdx
        for cond in condition_list:
            match_mask &= evaluate_join(device.type, tensors_left, tensors_right, cond, CS, CR)

        # Now compose outputs depending on join type
        matched_positions = torch.nonzero(match_mask, as_tuple=False).view(-1)
        left_matched = CS[matched_positions] if matched_positions.numel() > 0 else torch.tensor([], dtype=torch.int64, device=device)
        right_matched = CR[matched_positions] if matched_positions.numel() > 0 else torch.tensor([], dtype=torch.int64, device=device)
    else:
        left_matched = torch.tensor([], dtype=torch.int64, device=device)
        right_matched = torch.tensor([], dtype=torch.int64, device=device)

    if type == 'inner':
        leftOutIdx = left_matched
        rightOutIdx = right_matched
    elif type == 'right-semi':
        # return unique right ids that matched
        rightOutIdx = torch.unique(right_matched) if right_matched.numel() > 0 else torch.tensor([], dtype=torch.int64, device=device)
        leftOutIdx = torch.tensor([], dtype=torch.int64, device=device)
    elif type == 'right-anti':
        # right rows that never matched
        matched_rights_all = torch.unique(right_matched) if right_matched.numel() > 0 else torch.tensor([], dtype=torch.int64, device=device)
        if matched_rights_all.numel() == 0:
            rightOutIdx = torch.arange(nR, device=device, dtype=torch.int64)
        else:
            mask_all = torch.ones(nR, dtype=torch.bool, device=device)
            mask_all[matched_rights_all] = False
            rightOutIdx = torch.nonzero(mask_all, as_tuple=False).view(-1)
        leftOutIdx = torch.tensor([], dtype=torch.int64, device=device)
    elif type == 'right-outer':
        # matched pairs plus unmatched rights appended as nulls on left side
        matched_rights_all = torch.unique(right_matched) if right_matched.numel() > 0 else torch.tensor([], dtype=torch.int64, device=device)
        mask_all = torch.ones(nR, dtype=torch.bool, device=device)
        if matched_rights_all.numel() > 0:
            mask_all[matched_rights_all] = False
        unmatched_rights = torch.nonzero(mask_all, as_tuple=False).view(-1)  # these will be null-left
        # outputs: existing matched pairs + unmatched rights (left: nulls)
        leftOutIdx = left_matched
        rightOutIdx = right_matched
        if unmatched_rights.numel() > 0:
            rightOutIdx = torch.cat([rightOutIdx, unmatched_rights])
            nulls = unmatched_rights.size(0)
        else:
            nulls = 0
    else:
        raise NotImplementedError("unhandled join type")

    select_function = lambda x, idx: x[idx]
    if type == 'right-outer' and leftOutIdx.numel() > 0:
        leftOutIdx = torch.cat(
        (leftOutIdx,
        torch.full((nulls, ), constants.null, dtype=torch.int64, device=device_name)
        ), dim=0)
        select_function = index_with_null

    result = [] 
    for k in join_output_keys:
        if k in tensors_left:
            if tensors_left[k].tensor.shape[0] > 0:
                result.append(select_function(tensors_left[k].tensor, leftOutIdx))
            else:
                result.append(torch.tensor([]))
        else:
            if tensors_right[k].tensor.shape[0] > 0:
                result.append(tensors_right[k].tensor[rightOutIdx])
            else:
                result.append(torch.tensor([]))
    
    if return_global_index:
        result.extend([select_function(left_global_idx.tensor, leftOutIdx), right_global_idx.tensor[rightOutIdx]])

    return result

def join_vortex(gpu_enable, tensor_group, args, leftcol, rightcol, condition_list, type,
                cpu_mem_pool, gpu_mem_pool, name):
    perf_logger().start(name)

    #### Print slice of both side's keys
    # for k in args['left group']:
    #     message_logger().debug("ROWS -- %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS -- %s %s", k, tensor_group[k].tensor[0:50])
    # for k in args['right group']:
    #     message_logger().debug("ROWS ++ %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS ++ %s %s", k, tensor_group[k].tensor[0:50])

    join_key_index = [leftcol, rightcol]
    tensor_groupings = [args['left group'], args['right group']]
    join_mask_map = {}
    join_input = {}
    join_out_types = []
    join_output = []
    
    for side in [LEFT, RIGHT]:
        tensor_group[join_key_index[side]].normalize()
        for used_col in args['materialized in']:
            if used_col in tensor_groupings[side]:
                join_input[used_col] = tensor_group[used_col]
                join_output.append(used_col)
                join_out_types.append(tensor_group[used_col].tensor.dtype)
                if 'mask map' in args and used_col in args['mask map']:
                    mask_id = args['mask map'].pop(used_col)  
                    join_mask_map[used_col] = mask_id
                    join_input[mask_id] = tensor_group[mask_id]
    
    def join_wrapper(columns):
        ### Recover all the arguments to pass into in-GPU join kernel
        tensor_inputs = [{}, {}]
        for side in [LEFT, RIGHT]:
            for k in tensor_groupings[side]:
                if k in columns:
                    tensor_inputs[side][k] = columns[k]

        return join_kernel_new(True, tensor_inputs[LEFT], tensor_inputs[RIGHT], join_key_index[LEFT], join_key_index[RIGHT], 
                            condition_list, join_output, False, type, torch.tensor([]), torch.tensor([]))

    pipe = InMemoryPipeline(
        input_columns=join_input,
        output_columns_type=join_out_types,
        operator=join_wrapper,
        mask_map=join_mask_map,
        cpu_mem_pool=cpu_mem_pool,
        gpu_mem_pool=gpu_mem_pool,
        name=f'{name} core pipe'
    )
    pipe.do_exchange(20_000_000)
    join_results = pipe.get_result()
    message_logger().debug(join_results)

    for i, k in enumerate(join_output):
        tensor_group[k] = Variable(join_results[i].tensor, 
                                                    tensor_group[k].tensor_type,
                                                    join_results[i].memory_ptr,
                                                    join_results[i].placement)
        
    # for k in args['left group']:
    #     message_logger().debug("ROWS -- %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS -- %s %s", k, tensor_group[k].tensor[0:50])
    # for k in args['right group']:
    #     message_logger().debug("ROWS ++ %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS ++ %s %s", k, tensor_group[k].tensor[0:50])

    perf_logger().stop(name)
    return next(iter(tensor_group.values())).tensor.size(0)


