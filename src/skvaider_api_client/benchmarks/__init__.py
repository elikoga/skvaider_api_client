from .parallel import ParallelBenchmark
from .batch_api import BatchApiBenchmark
from .mixed import MixedBenchmark
from .sustained import SustainedBenchmark
from .all import AllBenchmark
from .base import BaseBenchmark

BENCHMARKS = [
    ParallelBenchmark,
    BatchApiBenchmark,
    MixedBenchmark,
    SustainedBenchmark,
    AllBenchmark,
]
