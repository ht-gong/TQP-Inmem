import torch
from typing import List, Dict
from dataclasses import dataclass

from utility.logger import perf_logger
from variable import Variable, VariableState


@dataclass
class ColumnMetaData:
    memory_ptr: int
    tensor: torch.Tensor
    cur_ind: int
    placement: VariableState


class InMemoryPipeline:
    def __init__(self, input_columns: Dict[int, Variable], output_columns_type: List[torch.dtype], operator,
                 cpu_mem_pool, gpu_mem_pool=None, div=None, chunk_size=20_000_000_000, pass_mask=False,
                 mask_map: Dict[int, int] = {}, name="InMemPipe"):

        self.__input_cols = input_columns
        self.__out_host = []
        self.__output_columns_typed = [torch.tensor([], dtype=t, device='cuda:0') for t in output_columns_type]
        self.__operator = operator
        self.__name = name
        self.__gpu_mem_pool = gpu_mem_pool
        self.__pass_mask = pass_mask

        self.__mask_map = {}
        cols = list(self.__input_cols.keys())
        for k, v in mask_map.items():
            assert v in cols, "mask col not passed in"
            self.__mask_map[cols.index(k)] = cols.index(v)

        for col in cols:
            assert self.__input_cols[col].tensor.is_cuda
            self.__input_cols[col].tensor = self.__input_cols[col].tensor.contiguous()

    def do_exchange(self, granularity: int):
        perf_logger().start(f"{self.__name}")
        outbuf = self.safe_apply_op([t.tensor for t in self.__input_cols.values()])
        perf_logger().stop(f"{self.__name}")
        torch.cuda.synchronize()
        for i in range(len(outbuf)):
            ptr, storage = self.__gpu_mem_pool.malloc_like(outbuf[i])
            storage[:] = outbuf[i]
            self.__out_host.append(ColumnMetaData(ptr, storage, -1, VariableState.GPU))
            outbuf[i] = None

    def safe_apply_op(self, input_tensors: List[torch.tensor]):
        if all(t.numel() == 0 for t in input_tensors):
            return [torch.tensor([], device='cuda') for _ in self.__output_columns_typed]
        input_dict = {}
        for (col_name, old_var), (i, tensor) in zip(self.__input_cols.items(), enumerate(input_tensors)):
            if i not in self.__mask_map.values():
                if i in self.__mask_map.keys():
                    input_dict[col_name] = Variable(tensor[input_tensors[self.__mask_map[i]]],
                                                    old_var.tensor_type)
                else:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type)
            else:
                if self.__pass_mask:
                    input_dict[col_name] = Variable(tensor, old_var.tensor_type)
                else:
                    input_tensors[i].tensor = None
        res = self.__operator(input_dict)
        if isinstance(res, torch.Tensor) and res.ndim == 0:
            res = [res]
        return list(res)

    def get_result(self):
        return self.__out_host
