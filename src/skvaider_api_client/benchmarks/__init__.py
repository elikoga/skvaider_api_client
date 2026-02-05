from .parallel import ParallelBenchmark
from .batch_api import BatchApiBenchmark
from .mixed import MixedBenchmark
from .base import BaseBenchmark

BENCHMARKS = [
    ParallelBenchmark,
    BatchApiBenchmark,
    MixedBenchmark
]
