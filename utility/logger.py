import logging
import time
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
from torch.profiler import profile, record_function, ProfilerActivity
import json
import os, sys
from datetime import datetime
from contextlib import nullcontext

# logging.getLogger().setLevel(logging.CRITICAL + 1) # Disable Root Logger

# Save original method
_orig_debug = logging.Logger.debug

def _debug_override(self, msg, *args, **kwargs):
    # If msg is a callable, only call it if DEBUG is enabled
    if callable(msg):
        if self.isEnabledFor(logging.DEBUG):
            text = msg()
            return _orig_debug(self, text, *args, **kwargs)
        else:
            return  # do nothing
    else:
        # msg is a normal string, let original handle it
        return _orig_debug(self, msg, *args, **kwargs)

logging.Logger.debug = _debug_override
_perf_logger = None
_datasize_logger = None
_message_logger = None
_torch_profiler = None

_mem_consumption = 0

def add_mem_consumption(n):
    global _mem_consumption
    _mem_consumption += n 

def get_mem_consumption():
    global _mem_consumption
    return _mem_consumption

def reset_mem_consumption():
    global _mem_consumption
    _mem_consumption = 0


def perf_logger():
    global _perf_logger
    if not _perf_logger:
        raise RuntimeError("Perf Logger Not set, run set_perf_logger first")   
    return _perf_logger

def message_logger():
    global _message_logger
    if not _message_logger:
        raise RuntimeError("Message Logger Not set, run set_message_logger first") 
    return _message_logger

def datasize_logger():
    global _datasize_logger
    if not _datasize_logger:
        raise RuntimeError("Datasize Logger Not set, run set_datasize_logger first")   
    return _datasize_logger


def set_perf_logger(filename=None, enable=True):
    global _perf_logger
    _perf_logger = PerformanceTracker()
    if not enable:
        _perf_logger.disable_handler()
    else: 
        _perf_logger.set_handlers(filename) 

def set_datasize_logger(filename=None):
    global _datasize_logger
    _datasize_logger = DatasizeTracker(filename)

def set_message_logger(filename=None, enable=True):
    global _message_logger
    _message_logger = setup_message_logger() 
    if not enable:
        disable_handler(_message_logger) 
    else:
        set_handlers(_message_logger, filename, 'debug')

def torch_profiler():
    global _torch_profiler
    if not _torch_profiler:
        raise RuntimeError("Torch Profiler Not set, run set_torch_profiler first") 
    return _torch_profiler

def set_torch_profiler(filename=None, enable=True, torch_profile_dir=None):
    global _torch_profiler 
    _torch_profiler = TorchProfiler(trace_dir=torch_profile_dir)
    if not enable:
        _torch_profiler.disable_handler()
    else: 
        _torch_profiler.set_handlers(filename) 

def default_formatter():
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return formatter

def set_handlers(
    logger: logging.Logger,
    dirname: str = None,
    prefix: str = ''
):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(default_formatter())
    logger.addHandler(stream_handler)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(dirname, f"{prefix}_{timestamp}.log")
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setFormatter(default_formatter())
        logger.addHandler(file_handler)
    
def disable_handler(
    logger: logging.Logger,
):
    logger.setLevel(logging.CRITICAL + 1)


def setup_performance_logger(
    name: str = "performance",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def setup_message_logger(
    name: str = "messaging",
    level: int = logging.DEBUG,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def setup_datasize_logger(
    name: str = "datasize",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def setup_profile_logger(
    name: str = "profiling",
    level: int = logging.INFO,
    fmt: str = "%(message)s",
) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class PerformanceTracker:
    """
    Tracks and logs performance metrics across multiple tags using a single logger.

    Features:
      - Manual timing: start(tag) / stop(tag)
      - Context manager: with tracker.time(tag): ...
      - Decorator: @tracker(tag)
      - Indexing: tracker[tag] -> {'count', 'total', 'avg'}
      - Reporting: tracker.report() logs all tag stats
    """
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or setup_performance_logger()
        self._stats = defaultdict(lambda: {'count': 0, 'total': 0.0})
        self._start_times = {}
        self._input_size = defaultdict(lambda: 0)
        self._enable = True

    def set_handlers(self, filename: str):
        set_handlers(self.logger, filename, 'perf')
    
    def disable_handler(self):
        disable_handler(self.logger)
        self._enable = False

    def start(self, tag: str):
        """Begin timing for given tag."""
        if not self._enable:
            return
        
        if tag in self._start_times:
            raise RuntimeError(f"Timing for '{tag}' already started.")
        self._start_times[tag] = time.perf_counter()

    def stop(self, tag: str) -> float:
        """End timing for tag, log and accumulate stats."""
        if not self._enable:
            return
        if tag not in self._start_times:
            raise RuntimeError(f"Timing for '{tag}' was not started.")
        elapsed = time.perf_counter() - self._start_times.pop(tag)

        return self.stop_and_record(tag, elapsed)

    def stop_and_record(self, tag: str, elapsed:float):
        stat = self._stats[tag]
        stat['count'] += 1
        stat['total'] += elapsed
        avg = stat['total'] / stat['count']
        self.logger.log(
            self.logger.level,
            f"[{tag}] elapsed: {elapsed:.6f}s "
            f"(count={stat['count']}, total={stat['total']:.6f}s, avg={avg:.6f}s)"
        )
        return elapsed

    def record(self, event, stream):
        if stream:
            event.record(stream)
        else:
            event.record()

    @contextmanager
    def time(self, tag: str, bytes=0):
        """Context manager for timing a code block under a tag."""
        self.start(tag)
        self._input_size[tag] += bytes
        try:
            yield
        finally:
            self.stop(tag)
    
    @contextmanager
    def time_in_stream(self, start_event, end_event, stream=None):
        """Context manager for timing a code block under a and stream."""
        self.record(start_event, stream)
        try:
            yield
        finally:
            self.record(end_event, stream)
    
    def time_after_sync(self, tag, start_event, end_event):
        return self.stop_and_record(tag, start_event.elapsed_time(end_event) * 1e-3)

    def __call__(self, tag: str):
        """Decorator for timing function calls under a tag."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.time(tag):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def report(self, filepath: str = "performance_stats.json") -> dict:
        """
        Serialize aggregated statistics for all tags into a JSON file.
        Returns the dict of stats written.
        """
        if not self._enable:
            return {}
        
        data = {}
        bytes_per_GB = 1e9
        for tag, stat in self._stats.items():
            count = stat['count']
            total = stat['total']
            avg = total / count if count else 0.0
            data[tag] = {'count': count, 'total': total, 'avg': avg, 'input GBs': self._input_size[tag] / bytes_per_GB}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        self.logger.log(self.logger.level, f"Statistics written to {filepath}")
        return data

    def __getitem__(self, tag: str):
        """Get stats dict for a tag: {count, total, avg}."""
        stat = self._stats.get(tag, {'count': 0, 'total': 0.0})
        count = stat['count']
        total = stat['total']
        avg = total / count if count else 0.0
        return {'count': count, 'total': total, 'avg': avg}


class TorchProfiler:
    def __init__(
        self,
        logger: logging.Logger=None,
        activities=None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        top_k: int = 5,
        trace_dir: str = None
    ):
        self.logger = logger or setup_profile_logger()
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.top_k = top_k
        self.profiler = None
        self._enable = True
        self.trace_dir = trace_dir

    @contextmanager
    def profile(self, tag):
        if not self.trace_dir:
            yield nullcontext(None)
            return
        self.tag = tag
        if self.profile_memory:
            self.record_shapes = True
            self.with_stack    = True
        with profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        ) as prof:
            yield prof

        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(self.trace_dir, f'{self.tag}_trace.json'))
        
        if self.profile_memory:
            self.logger.info("Top %d CUDA operations by self memory consumption:", self.top_k)
            cuda_table = prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=self.top_k
            )
            for line in cuda_table.splitlines():
                self.logger.info(line)

            self.logger.info("Top %d CUDA operations by memory consumption:", self.top_k)
            cuda_table = prof.key_averages().table(
                sort_by="cuda_memory_usage", row_limit=self.top_k
            )
            for line in cuda_table.splitlines():
                self.logger.info(line)
    def set_handlers(self, filename: str):
        set_handlers(self.logger, filename, 'profile')
    
    def disable_handler(self):
        disable_handler(self.logger)
        self._enable = False


class DatasizeTracker:
    def __init__(self, output_dir=str):
        self.logger = setup_datasize_logger()
        self._sum = defaultdict(lambda: defaultdict(int))
        self._output_dir = output_dir
        self._mem_util = []
        self._mem_ext_frag = []
    
    def set_operator(self, name: str):
        self._name = name

    def mem_stats(self, mem_pool):
        self._mem_util.append(mem_pool.get_utilization())
        self._mem_ext_frag.append(mem_pool.get_external_frag())

    def record(self, tag: str, size: int):
        """End timing for tag, log and accumulate stats."""
        if not self._output_dir:
            return

        self.logger.log(
            self.logger.level,
            f"[{self._name}][{tag}] Transfer: {size / (10 ** 9):.3f} GB"
        )
        self._sum[self._name][tag] += size

    def defaultdict_to_dict(self, d):
        if isinstance(d, defaultdict):
            d = {k: self.defaultdict_to_dict(v) for k, v in d.items()}
        return d

    def report(self, filename) -> dict:
        if not self._output_dir:
            return {}
        # write to file
        res = self.defaultdict_to_dict(self._sum)
        res['mem_util'] = self._mem_util
        res['mem_ext_frag'] = self._mem_ext_frag
        
        with open(os.path.join(self._output_dir, filename), 'w') as f:
            json.dump(res, f, indent=4)
        self.logger.log(self.logger.level, f"Statistics written to {self._output_dir}")

