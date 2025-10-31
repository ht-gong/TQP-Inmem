# TQP-Vortex
This is an re-implementation of the Microsoft Tensor Query Processing platform, and a fork of the on-going project to make TQP support out-of-GPU memory queries.

A large part of the original code volume is for out-of-memory processing. This fork removes the out-of-memory processing part to support only in-memory workloads as an experimental baseline.

The structure of the repo is approximately:
```
TQP-Vortex
    ├── cmake          : CMake files for building custom modules
    ├── experiments    : Graph plotting, micro-benchmarks        
    ├── IO             : Module for handling CPU-GPU I/O
    ├── memory         : Custom memory allocator modules for PyTorch that handles GPU-side memory allocation and Pinned-memory allocation
    ├── operators      : Database operators (Join, Filter, Aggregation, Project, ...) implemented using PyTorch
    ├── queries        : Query files and parsed intermediate forms (from Spark query plan)
    ├── tests          : Tests for the system
    ├── TQPlib         : Custom modules after compilation gets saved to this folder
    ├── utility        : Common helpers
    ├── Vortex         : External module that experiments with multi-PCIE speedups with multi-GPUs
    ├── parsing.py     : Invokes Spark's query plan generator and then parses queries into a format that our system can run
    └── main.py        : Main entrypoint for the system
```
## Environment Setup
Choose your favorite python environment manager - we use Conda as an example.

### Conda
Create a Conda Environment and install torch.

```
conda create -y -n torch126 python=3.12 cmake ninja gcc_linux-64=11 gxx_linux-64=11   cuda-toolkit=12.6 -c conda-forge
conda activate torch126
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```


### TPCH-DBGEN with Tensors

We wrote torch-native dbgen linked here: https://github.com/hfhongzy/tpch-dbgen-tensors/tree/main

Clone the repo and use the following instructions to configure the tpch-dbgen repo under its root directory.

``` 
mkdir -p build/ data/
cd build/
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux
make -j
./dbgen -s 1
```
Substitute the -s flag with dbgen with the desired scale factor to generate the corresponding set of data tables in tensor form.
The tensors will be generated under ./data from the tpch-dbgen repo home directory

### Make TQP Custom Modules
From the TQP-Vortex home directory, run

```
cmake -S . -B build/ -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux
cmake --build build -j
pip install --no-cache-dir -v .
pip intall -r requirements.txt
```
(HIP / Nvidia envs should be automatically detected)

## Running the System
First, modify ``config.json`` to point to ``${dbgen_root_dir/data}`` 

e.g. in config.json:
```
"tensors path": ${dbgen_root_dir/data}
```
*Need to substitute dbgen_root_dir with the directory where dbgen is cloned at.*

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
python3 main.py --SF 10 --q 1 2 6 18 21  --warmup_iter 2 --time_output_file ./time.csv --no_debug
``` 

We obtain following numbers:

| qid | our system (s) |
| :-: | ---------------: |
|  1  |         0.091427 |
|  2  |         0.034542 |
|  6  |         0.005435 |
|  9  |         0.121054 |
|  18 |         0.078482 |


One note is that we calculate the runtim numbers by subtracting the total runtime by the result output time (writing to screen/file) and the substracting the scan time (scanning disk to GPU).
