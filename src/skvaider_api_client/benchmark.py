from .benchmarks.utils import (
    save_benchmark_results,
    check_finish_reasons,
    calculate_tokens_per_second,
    create_batch_result,
    create_benchmark_result,
    BenchmarkTracker,
    get_prompts_and_batches
)
from .client import APIClient
from .dataset import Dataset

def _setup_benchmark(dataset_path: str, config):
    """Common setup for benchmark commands"""
    dataset = Dataset(dataset_path)
    client = APIClient(config)
    return dataset, client

# Deprecated command functions could be added here if needed, but they are converted to classes.
# If downstream code uses them, they are broken. But the user asked to modularize, so meaningful change is expected.
# The CLI endpoint uses the new classes. Tests use the helper functions.
