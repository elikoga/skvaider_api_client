import json
from ..dataset import Dataset

def save_benchmark_results(output_path: str, data: dict):
    """Save benchmark results to a JSON file"""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved benchmark results to {output_path}")


def check_finish_reasons(responses: list[dict], batch_idx: int, expected_reason: str = "length") -> bool:
    """Check if all responses finished with expected reason. Warns on mismatch instead of failing."""

    def check_single_response(resp: dict) -> bool:
        return all(choice["finish_reason"] == expected_reason for choice in resp["choices"])

    all_match = all(check_single_response(resp) for resp in responses)
    if not all_match:
        mismatched = [resp for resp in responses if not check_single_response(resp)]
        print(f"  WARNING: Not all completions finished with reason '{expected_reason}' in batch {batch_idx}")
        print(f"  Mismatched responses: {len(mismatched)} out of {len(responses)}")
    return all_match


def calculate_tokens_per_second(tokens: int, elapsed_time: float) -> float:
    """Calculate tokens per second, handling zero time"""
    return tokens / elapsed_time if elapsed_time > 0 else 0.0


def create_batch_result(batch_idx: int, batch_size: int, tokens: int, elapsed_time: float, **extra) -> dict:
    """Create a standardized batch result dictionary"""
    result = {
        "batch_idx": batch_idx,
        "batch_size": batch_size,
        "tokens": tokens,
        "time": elapsed_time,
        "tokens_per_second": calculate_tokens_per_second(tokens, elapsed_time),
    }
    result.update(extra)
    return result


def create_benchmark_result(
    batch_size: int, total_prompts: int, total_tokens: int, total_time: float, batch_results: list[dict]
) -> dict:
    """Create a standardized benchmark result summary"""
    return {
        "batch_size": batch_size,
        "total_prompts": total_prompts,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "average_tokens_per_second": calculate_tokens_per_second(total_tokens, total_time),
        "batch_results": batch_results,
    }

def get_prompts_and_batches(dataset: Dataset, total_prompts: int, batch_size: int):
    """Get prompts (with replication if needed) and create batches"""
    prompts = dataset.get_n_samples(total_prompts)
    if total_prompts > len(dataset.data):
        print(f"  Note: Dataset replicated to create {total_prompts} prompts")
    batches = Dataset.create_batches(prompts, batch_size)
    return prompts, batches


class BenchmarkTracker:
    """Track tokens, time, and batch results during benchmarking"""

    def __init__(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.batch_results = []

    def add_batch_result(self, batch_idx: int, batch_size: int, tokens: int, elapsed_time: float, **extra):
        """Add a batch result and update totals"""
        self.total_tokens += tokens
        self.total_time += elapsed_time
        result = create_batch_result(batch_idx, batch_size, tokens, elapsed_time, **extra)
        self.batch_results.append(result)
        return result
