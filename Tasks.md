# Tasks

This section outlines what you should complete for your starter tasks, and the deliverables we expect.

This task is meant to blend research-like exploration with a manageable amount of engineering, and a report of your findings.

For the task, you can pick **one** of the following 3 options:
- Work on one of the major "exploration" tasks down below.
- Implement any two from the "improvement" tasks below.
- Build your own end-to-end hypothesis and verification, outlined in the "DIY" section below.

## Exploration

Choose 1 from this section:

### High-performance Hashing 

In traditional database operator implementations, the hashing function is often central to the performance.

Currently, the "hashing" in TQP is only a blend of PyTorch operator calls and is not very performance-tuned to the state-of-the-art implementations.

For this task, you should 
- Identify *where* exactly in the operators we are using a hashing-like functionality.
     - `operators/` implementation is a good starting point.
- Investigate current/past literature to get a background of how high-performance hashing is done on different hardware.
- Build your own proof-of-concept hashing implementation (taken from literature or by your own invention), to show a different way of using the hashing from our current codebase, and demonstrate potential tradeoffs/whether it works well.
- Good stress queries: join-heavy TPC-H (Q3, Q5, Q9, Q10) and group-by-heavy (Q1, Q13, Q18). Use `main.py --q <id>` with `perf_logger` output to isolate per-operator times.

### Memory Optimization

The current TQP implementation is bound by available GPU memory. When run on top of a 24GB GPU, we can barely reach scale-factor 10 on TPC-H runs or less.

There are PyTorch operations that are very memory-consuming and generate large intermediates. Can we optimize these Torch calls? Or are there other ways to gracefully deal with memory scarcity on the GPU without modifying Torch code?

For this task, you should 
- Identify *where* exactly is the bottleneck and expensive memory calls in Torch.
    - `operators/aggregate.py` and `operators/hashjoin.py` are good starting points.
- Either 
    - investigate Torch optimization options to overcome these calls.
    - examine existing literature to see how systems deal with large memory consumptions.
- Build your own proof-of-concept memory optimization implementation, and demonstrate potential tradeoffs/whether it works well.

- Instrumentation: `torch.cuda.memory_stats()`, `torch.cuda.max_memory_allocated()`, and the PyTorch memory profiler (`torch.profiler.profile(profile_memory=True)`). `main.py` already exposes `--torch_profile` and `--datasize_profile`.
    — see the argparse block at `main.py:314-324`.
- Good stress queries: Q18 (large join + group-by), Q9 (multi-way join), Q21 (self-joins). On a 24 GiB card these are the first queries to OOM.

### Out-of-Core Implementation

The current TQP system is hard-limited by available GPU memory as the processing region. What if the data lives on the CPU and can be *streamed* to the GPU, so that we are not limited by the GPU memory any more?

For this task, you should 
- Identify *where* and *how* we can parition data to move to the GPU.
    - `operators/scan.py:22-48` is the ingress point. Whole column tensors are loaded from disk via `torch.jit.load` and then copied into GPU memory via `mem_pool.malloc_like(...)`.
- Build a design plan for how to alter the current system to do so.
- Build your own proof-of-concept streaming component that overcomes the memory limit for *a few* specific queries, and demonstrate potential tradeoffs/whether it works well.
- Good first queries to target: Q6 (single-table scan + filter + sum), Q1 (large group-by over `lineitem`), Q14.

## Improvement

These are focused more on improving the individual components. Choose any 2 from this section.

### Aggregation Performance 

The aggregation operator is known to be slow for TQP. 
- Investigate its bottlenecks/scaling properties.
- Gain insight into why it is slow.
- Implement proof of concept improvements (maybe use something other than PyTorch?) to see whether there are benefits.

### Memory Allocator Performance

The memory allocator can be a bottleneck for PyTorch on allocation-heavy workloads. Our current `GPUMemory` / `PinnedMemory` in `IO/pinned_mem.py` is a pure-Python block allocator on top of a single `torch.empty(...)` slab.

- Profile allocator traffic during query execution — which operators trigger the most malloc/free, and what are the request-size distributions?
- Compare the current Python block allocator against alternatives (PyTorch's native caching allocator, a C++ reimplementation of the current pool, the CUDA driver's `cuMemPool` API, a slab/arena design).
- Implement a proof-of-concept alternative and measure the end-to-end effect on representative TPC-H queries.


### Data Materialization Overhead

In our current operator implementations, there are multiple sections dedicated to transforming the data and 'materializing' it to the GPU main memory.

- Identify where materialization happens — `InMemoryPipeline` usage inside the operators, intermediate tensor allocations, and type conversions in `conversion.py` are good starting points.
- Investigate whether these steps can be deferred, fused across operator boundaries, or expressed as tensor views rather than physical copies.
- Implement a proof-of-concept change for one or two representative queries and report the effect on runtime and peak memory use.


## DIY

If none of the Exploration or Improvement tasks captures what interests you, propose your own.

A good DIY task has:
- **A hypothesis** — a specific, testable general claim about TQP's behavior (e.g. "operator X is bandwidth-bound because Y"), not just a topic area.
- **A detailed design** — how you will confirm or refute it, and what measurements / experiments you'll run.
- **A proof-of-concept code implementation** — code that demonstrates the finding, even if small in scope.

# Deliverables

Whichever track you pick, we expect the following 2 artifacts.

## Code

- A branch ready for review, runnable from a fresh checkout following `Setup.md`.

## Write-up

A short report (roughly 2-4 pages). We suggest that you cover:

- **Motivation** — which task you picked and the specific hypothesis or goal.
- **Background & Approach** — are there existing literature you reviewed, what you changed, what you tried and rejected.
- **Results** — the experiments you ran, on what hardware, at what scale factor(s). The results do not have to be very polished or all positive, but you should try to reason why they happen.
- **Tradeoffs and next steps** — what your change costs (memory, code complexity, portability), what it doesn't handle.
- **Reproduction** — the exact commands to regenerate your numbers from a fresh checkout, plus GPU / driver / CUDA / PyTorch version.
