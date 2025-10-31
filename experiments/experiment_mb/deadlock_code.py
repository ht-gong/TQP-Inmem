import torch

# 1) Ensure cuda:2 exists
if torch.cuda.device_count() <= 2:
    raise RuntimeError("Need at least 3 CUDA devices to use cuda:2")

# 2) Select device 2
torch.cuda.set_device(2)

# 3) Configure a ~5 min spin
sleep_secs = 50 * 60               # 300 s
gpu_clock_hz = 1e9                # ~1 GHz core clock (adjust if you know yours)
cycles_to_sleep = int(gpu_clock_hz * sleep_secs)

# 4) Create a dedicated non‐default stream on cuda:2
stream_sleep = torch.cuda.Stream(device='cuda:2')

# 5) Launch the busy‐wait kernel into that stream
with torch.cuda.stream(stream_sleep):
    torch.cuda._sleep(cycles_to_sleep)   # enqueues the spin on stream_sleep

# — meanwhile the default stream is free to do other work —
print("GPU sleep is running asynchronously on stream_sleep...")

# 6) When you’re ready to wait for the spin to finish:
stream_sleep.synchronize()
print("✅ 5 min GPU sleep on cuda:2 (stream_sleep) is done.")
