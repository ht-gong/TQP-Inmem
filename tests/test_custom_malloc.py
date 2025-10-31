import torch, sys
# torch_alloc_pressure_min.py
import  random, math, gc

from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
sys.path.insert(0, "/home/jovyan/workspace/TQP-Vortex/memory/build")
import cuda_mem_pool as cmp
# Path to the freshly built .so
path = "/home/jovyan/workspace/TQP-Vortex/memory/build/libcustom_allocator.so"

alloc = CUDAPluggableAllocator(
    path_to_so_file=path,
    alloc_fn_name="pa_cuda_malloc",
    free_fn_name="pa_cuda_free",
)
change_current_allocator(alloc)  # must be called before any CUDA tensors exist

cmp.init(75*1024**3, 75*1024**3 // 1000)
# Optional tuning via env:
#   PA_CUDA_POOL_CAP_MB=4096  # 4 GiB per-device soft cap
#   PA_CUDA_VERBOSE=1         # prints allocator events
#   PA_CUDA_MIN_BIN_BYTES=4096


# torch_alloc_pressure_cap.py

# ---- Tunables ----
DEVICE    = "cuda"
SEED      = 0
MAX_LIVE  = 2000      # <-- how many arrays live at one time (hard cap)
MIN_KB    = 4
MAX_MB    = 128
TOUCH_PROB = 0.0      # chance to write into a new tensor
# -------------------

assert torch.cuda.is_available(), "CUDA not available"

rng = random.Random(SEED)
device = torch.device(DEVICE)

# Warmup (your sanity check)
x = torch.empty((1024, 1024), device=device)
del x
z = torch.empty((1024, 1024), device=device)
y = torch.empty((1024, 1024), device=device)
del z, y

live = []          # list of tensors
ELSZ = 4           # float32 bytes

def alloc_one():
    want_bytes = int(2 ** rng.uniform(
        math.log2(MIN_KB * 1024),
        math.log2(MAX_MB * 1024 * 1024)
    ))
    n = max(1, want_bytes // ELSZ)     # 1D shape for simplicity
    while True:
        try:
            t = torch.empty((n,), device=device, dtype=torch.float32)
            if rng.random() < TOUCH_PROB:
                t.fill_(1)
            live.append(t)
            return
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and live:
                # Free one random live tensor and retry
                i = rng.randrange(len(live))
                del live[i]
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise

try:
    while True:
        if len(live) < MAX_LIVE:
            alloc_one()
        else:
            i = rng.randrange(len(live))
            # 50% free only, 50% free+allocate (churn)
            if rng.random() < 0.5:
                del live[i]
                gc.collect()
            else:
                del live[i]
                gc.collect()
                alloc_one()
except KeyboardInterrupt:
    pass
finally:
    live.clear()
    gc.collect()
    torch.cuda.empty_cache()

