
import time
from utility.import_utils import load_vortex_lib
vortex = load_vortex_lib()
# import vortex
from utility.tensor_utils import show_tensor_usage
from utility.logger import datasize_logger, message_logger
import torch
import sys
import TQPlib.zero_copy_reader as zero_copy_reader

print("Intializing and pre-warming Vortex exchange...")
_granularity = 20_000_000
# _exchange = vortex.Exchange(_granularity)
_exchange = 0
_enable_naive=False

class NaiveExchange:
    def __init__(self, granularity):
        self.granularity = granularity
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    def launch(self, dstDevice, srcHost, dstHost, srcDevice, mask_info = None, rearrange_info = None):
        assert isinstance(dstDevice, list) and isinstance(srcHost, list) and len(dstDevice) == len(srcHost), "Invalid host-to-device arguments."
        assert isinstance(dstHost, list) and isinstance(srcDevice, list) and len(dstHost) == len(srcDevice), "Invalid device-to-host arguments."
        assert not mask_info or (isinstance(mask_info, list) and len(mask_info) == len(srcHost))
        assert not rearrange_info or (isinstance(rearrange_info, list) and len(rearrange_info) == len(srcHost))
        for h, d in zip(srcHost + srcDevice, dstDevice + dstHost):
            assert h.device != d.device, "wrong device."
            assert h.dtype == d.dtype, f"wrong dtype. {h.dtype} != {d.dtype}."
        
        # self.__copy_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.h2d_stream):
            for i in range(len(dstDevice)):
                if mask_info and mask_info[i][0] is not None:
                    datasize_logger().record('Zero Copy CPU In', srcHost[i].numel() * srcHost[i].element_size())
                    dstDevice[i].resize_((mask_info[i][1], *dstDevice[i].shape[1:])) # Resize for zero-copy
                    zero_copy_reader.zero_copy_mask(srcHost[i], mask_info[i][0], dstDevice[i])
                elif rearrange_info and rearrange_info[0] is not None:
                    datasize_logger().record('Zero Copy CPU In', srcHost[i].numel() * srcHost[i].element_size())
                    zero_copy_reader.zero_copy_rearrange(srcHost[i], rearrange_info[0], dstDevice[i])
                else:
                    assert dstDevice[i].shape == srcHost[i].shape, f"wrong shape. {dstDevice[i].shape} != {srcHost[i].shape}"
                    datasize_logger().record('Naive CPU In', srcHost[i].numel() * srcHost[i].element_size())
                    dstDevice[i].copy_(srcHost[i], non_blocking=True)

        with torch.cuda.stream(self.d2h_stream):
            for i in range(len(dstHost)):
                assert dstHost[i].shape == srcDevice[i].shape, f"wrong shape. {dstHost[i].shape} != {srcDevice[i].shape}"
                datasize_logger().record('Naive CPU Out', srcDevice[i].numel() * srcDevice[i].element_size())
                dstHost[i].copy_(srcDevice[i], non_blocking=True)

    def sync(self):
        self.d2h_stream.synchronize()
        self.h2d_stream.synchronize()

def set_exchange_to_naive():
    global _granularity
    global _enable_naive
    global _exchange
    _exchange = NaiveExchange(_granularity)
    _enable_naive = True

def exchange(granularity: int=20_000_000):
    global _exchange
    global _granularity
    global _enable_naive
    if not _exchange or _granularity != granularity:
        print("Re-Initializing and pre-warming Exchange...")
        del _exchange
        if _enable_naive:
            _exchange = NaiveExchange(granularity)
        else:
            _exchange = vortex.Exchange(granularity)
        
        _granularity = granularity
    return _exchange


    