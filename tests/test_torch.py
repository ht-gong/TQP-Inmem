import torch
import time

# Set parameters: a tensor with 1 million integers and many duplicates.
size = 5_000_000_00
data = torch.randint(0, 1000, (size,), device='cuda:0')

# To ensure that both methods return the same result,
# sort the tensor so that duplicates are grouped consecutively.

# Define the method using torch.unique.
def unique_method(x):
    # torch.unique returns sorted unique elements.
    return torch.unique(x)

# Define the method using torch.unique_consecutive.
def unique_consecutive_method(x):
    # Assumes that duplicate elements occur consecutively.
    x_sorted, _ = torch.sort(x)
    return torch.unique_consecutive(x_sorted)

# Warm-up iterations to remove any one-time overhead.
_ = unique_method(data)
_ = unique_consecutive_method(data)

# Correctness verification: both should yield identical outputs because 'data_sorted' is sorted.
unique_result = unique_method(data)
unique_consecutive_result = unique_consecutive_method(data)
if torch.equal(unique_result, unique_consecutive_result):
    print("Correctness verification passed: Both methods produce the same output.")
else:
    print("Error: The two methods yield different outputs.")

# Helper function to benchmark a method by timing a number of repeated runs.
def benchmark(func, x, repeats=100):
    start = time.time()
    for _ in range(repeats):
        _ = func(x)
    end = time.time()
    return (end - start) / repeats

# Run the benchmarks.
repeats = 10
time_unique = benchmark(unique_method, data, repeats)
time_unique_consecutive = benchmark(unique_consecutive_method, data, repeats)

print(f"Average execution time using torch.unique             : {time_unique*1e3:.4f} ms per run")
print(f"Average execution time using torch.unique_consecutive   : {time_unique_consecutive*1e3:.4f} ms per run")
