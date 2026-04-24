# TQP-Inmem
This is an re-implementation of the Microsoft Tensor Query Processing platform ([paper](https://www.vldb.org/pvldb/vol15/p2811-he.pdf)), and a fork of the on-going project to make TQP support out-of-GPU memory queries.

A large part of the original code volume is for out-of-memory processing. This fork removes the out-of-memory processing part to support only in-memory workloads as an experimental baseline.

The structure of the repo is approximately:
```
TQP-Inmem
    ├── IO               : In-memory pipeline and pure-Python pinned/GPU memory pools (block-based allocator on top of torch.empty)
    ├── operators        : Database operators (Join, Filter, Aggregation, Project, ...) implemented using PyTorch
    ├── queries          : Query files and parsed intermediate forms (from Spark query plan)
    ├── tests            : Tests for the system
    ├── utility          : Common helpers (logger, tensor utilities)
    ├── config.json      : Points to the TPC-H tensor data directory
    ├── constants.py     : System-wide constants
    ├── conversion.py    : Type conversion helpers (dates, floats, strings)
    ├── expression.py    : SQL expression parser for Spark plan expressions
    ├── main.py          : Main entrypoint for the system
    ├── parsing.py       : Invokes Spark's query plan generator and then parses queries into a format that our system can run
    ├── requirements.txt : Python dependencies
    ├── test_main.py     : End-to-end correctness tests
    └── variable.py      : Variable / tensor-type wrappers
```
## Environment Setup

See [`Setup.md`](./Setup.md) for the conda environment, TPC-H data generation, and Python dependency steps.

Follow-up work and known rough edges are tracked in [`Tasks.md`](./Tasks.md).

## Running the System

Make sure `config.json`'s `tensors path` points to the tpch-dbgen `data/` directory (see `Setup.md`).

The system is runnable via 

```
python3 main.py --SF {scale_factor} --q {TPC-H query no.}  --warmup_iter {warmup iterations} --time_output_file {directory to file to save runtime measurements} [--no_debug]
```

Also, a basic correctness test is provided via

``python3 main.py --test``

## Evaluation and Results

On a Nvidia A100 - 80GB card, Using the command 
```
python3 main.py --SF 1 --warmup_iter 2 --time_output_file ./time.csv --no_debug
```

We obtain the following results. We also provide a comparison with the original TQP paper's numbers.
| Query | out_system (s) | TQP reported(s) | Speedup (TQP / our_system) |
| :---- | ---------------: | -------: | ----------------------------: |
| Q1    |         0.016011 |    0.026 |                         1.62× |
| Q2    |         0.019191 |    0.028 |                         1.46× |
| Q3    |         0.011614 |    0.024 |                         2.07× |
| Q4    |         0.007643 |    0.018 |                         2.36× |
| Q5    |         0.016295 |    0.042 |                         2.58× |
| Q6    |         0.003077 |    0.002 |                         0.65× |
| Q7    |         0.022092 |    0.035 |                         1.59× |
| Q8    |         0.021916 |    0.039 |                         1.78× |
| Q9    |         0.023115 |    0.092 |                         3.98× |
| Q10   |         0.032032 |    0.052 |                         1.62× |
| Q11   |         0.012422 |    0.009 |                         0.72× |
| Q12   |         0.010219 |    0.021 |                         2.06× |
| Q13   |         0.038741 |    0.136 |                         3.51× |
| Q14   |         0.007072 |    0.005 |                         0.71× |
| Q15   |         0.012898 |      N/A |                             — |
| Q16   |         0.021118 |    0.301 |                        14.26× |
| Q17   |         0.017744 |    0.051 |                         2.87× |
| Q18   |         0.022537 |    0.048 |                         2.13× |
| Q19   |         0.014386 |    0.036 |                         2.50× |
| Q20   |         0.012795 |    0.041 |                         3.20× |
| Q21   |         0.030219 |    0.151 |                         5.00× |
| Q22   |         0.009958 |    0.010 |                         1.00× |


Using
```
python3 main.py --SF 10 --q 1 2 6 9 18  --warmup_iter 2 --time_output_file ./time.csv --no_debug
``` 

We obtain following numbers:

| qid | our system (s) |
| :-: | ---------------: |
|  1  |         0.091427 |
|  2  |         0.034542 |
|  6  |         0.005435 |
|  9  |         0.121054 |
|  18 |         0.078482 |


One note is that we calculate the runtime numbers by subtracting the total runtime by the result output time (writing to screen/file) and then substracting the scan time (scanning from disk -> GPU).
