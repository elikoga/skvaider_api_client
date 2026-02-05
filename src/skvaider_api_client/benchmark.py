import asyncio
import json
import time

from .client import APIClient
from .config import Config
from .dataset import Dataset


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


def _setup_benchmark(dataset_path: str, config: Config):
    """Common setup for benchmark commands"""
    dataset = Dataset(dataset_path)
    client = APIClient(config)
    return dataset, client


def _get_prompts_and_batches(dataset: Dataset, total_prompts: int, batch_size: int):
    """Get prompts (with replication if needed) and create batches"""
    prompts = dataset.get_n_samples(total_prompts)
    if total_prompts > len(dataset.data):
        print(f"  Note: Dataset replicated to create {total_prompts} prompts")
    batches = Dataset.create_batches(prompts, batch_size)
    return prompts, batches


def _save_benchmark_results_with_summary(output_path: str, output_data: dict, summary_header: str, summary_formatter):
    """Save results and print summary"""
    save_benchmark_results(output_path, output_data)
    print(f"\n=== {summary_header} ===")
    print(f"Model: {output_data['model']}")
    summary_formatter(output_data)


async def _execute_parallel_batches(
    client: APIClient, model_id: str, batches: list[list[str]], max_tokens: int, tracker: BenchmarkTracker
):
    """Execute batches using parallel chat/completions requests"""
    for batch_idx, batch in enumerate(batches):
        start_time = time.time()
        tasks = [client.get_completion(model_id, prompt, max_tokens) for prompt in batch]
        responses = await asyncio.gather(*tasks)
        batch_time = time.time() - start_time

        check_finish_reasons(responses, batch_idx)
        batch_tokens = sum(resp["usage"]["completion_tokens"] for resp in responses)
        batch_result = tracker.add_batch_result(batch_idx, len(batch), batch_tokens, batch_time)
        print(
            f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {batch_result['tokens_per_second']:.2f} tok/s"
        )


async def _execute_batch_api_batches(
    client: APIClient, model_id: str, batches: list[list[str]], max_tokens: int, tracker: BenchmarkTracker
):
    """Execute batches using /completions batch API"""
    for batch_idx, batch in enumerate(batches):
        start_time = time.time()
        response = await client.get_batch_completion(model_id, batch, max_tokens)
        batch_time = time.time() - start_time

        check_finish_reasons([response], batch_idx)
        num_choices = len(response["choices"])
        tokens_per_choice = response["usage"]["completion_tokens"]
        batch_tokens = tokens_per_choice * num_choices
        batch_result = tracker.add_batch_result(batch_idx, len(batch), batch_tokens, batch_time)
        print(
            f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {batch_result['tokens_per_second']:.2f} tok/s"
        )


async def _execute_mixed_parallel_batch(
    client: APIClient,
    model_id: str,
    batches: list[list[str]],
    requests_at_once: int,
    max_tokens: int,
    tracker: BenchmarkTracker,
):
    """Execute using mixed parallel and batch API approach"""
    for batch_idx in range(0, len(batches), requests_at_once):
        current_batches = batches[batch_idx : batch_idx + requests_at_once]
        start_time = time.time()
        tasks = [client.get_batch_completion(model_id, batch, max_tokens) for batch in current_batches]
        responses = await asyncio.gather(*tasks)
        batch_time = time.time() - start_time

        check_finish_reasons(responses, batch_idx // requests_at_once)
        batch_tokens = sum(resp["usage"]["completion_tokens"] * len(resp["choices"]) for resp in responses)
        batch_result = tracker.add_batch_result(
            batch_idx // requests_at_once,
            len(current_batches),
            batch_tokens,
            batch_time,
            requests=len(current_batches),
        )
        print(
            f"  Requests {batch_idx // requests_at_once + 1}/{(len(batches) + requests_at_once - 1) // requests_at_once}: {batch_tokens} tokens in {batch_time:.2f}s = {batch_result['tokens_per_second']:.2f} tok/s"
        )


async def benchmark_batch_command(
    config: Config,
    model_id: str,
    dataset_path: str,
    max_tokens: int,
    batch_sizes: list[int],
    output_path: str,
):
    """Benchmark using /completions endpoint with prompt list batching"""
    dataset, client = _setup_benchmark(dataset_path, config)

    print(f"Benchmarking model (BATCH API): {model_id}")
    print(f"Base dataset size: {len(dataset.data)} prompts")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Testing batch sizes: {batch_sizes}\n")

    results = []
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        prompts, batches = _get_prompts_and_batches(dataset, len(dataset.data), batch_size)
        tracker = BenchmarkTracker()
        await _execute_batch_api_batches(client, model_id, batches, max_tokens, tracker)

        result = create_benchmark_result(
            batch_size, len(prompts), tracker.total_tokens, tracker.total_time, tracker.batch_results
        )
        results.append(result)
        print(
            f"  Overall: {tracker.total_tokens} tokens in {tracker.total_time:.2f}s = {result['average_tokens_per_second']:.2f} tok/s\\n"
        )

    output_data = {
        "model": model_id,
        "max_tokens": max_tokens,
        "method": "batch_api",
        "timestamp": time.time(),
        "results": results,
    }

    def print_summary(data):
        print("Batch Size | Avg Tokens/Second")
        print("-" * 35)
        for result in data["results"]:
            print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")

    _save_benchmark_results_with_summary(output_path, output_data, "SUMMARY (BATCH API)", print_summary)


async def benchmark_command(
    config: Config,
    model_id: str,
    dataset_path: str,
    max_tokens: int,
    batch_sizes: list[int],
    output_path: str,
):
    dataset, client = _setup_benchmark(dataset_path, config)

    print(f"Benchmarking model (PARALLEL): {model_id}")
    print(f"Base dataset size: {len(dataset.data)} prompts")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Testing batch sizes: {batch_sizes}\n")

    results = []
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        prompts, batches = _get_prompts_and_batches(dataset, len(dataset.data), batch_size)
        tracker = BenchmarkTracker()
        await _execute_parallel_batches(client, model_id, batches, max_tokens, tracker)

        result = create_benchmark_result(
            batch_size, len(prompts), tracker.total_tokens, tracker.total_time, tracker.batch_results
        )
        results.append(result)
        print(
            f"  Overall: {tracker.total_tokens} tokens in {tracker.total_time:.2f}s = {result['average_tokens_per_second']:.2f} tok/s\n"
        )

    output_data = {
        "model": model_id,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
        "results": results,
    }

    def print_summary(data):
        print("Batch Size | Avg Tokens/Second")
        print("-" * 35)
        for result in data["results"]:
            print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")

    _save_benchmark_results_with_summary(output_path, output_data, "SUMMARY", print_summary)


async def benchmark_mixed_command(
    config: Config,
    model_id: str,
    dataset_path: str,
    max_tokens: int,
    completions_per_request: int,
    requests_at_once: int,
    output_path: str,
):
    dataset, client = _setup_benchmark(dataset_path, config)

    print(f"Benchmarking model (MIXED): {model_id}")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Completions per request: {completions_per_request}")
    print(f"Requests at once: {requests_at_once}\n")

    total_prompts = completions_per_request * requests_at_once
    prompts, batches = _get_prompts_and_batches(dataset, total_prompts, completions_per_request)
    tracker = BenchmarkTracker()
    await _execute_mixed_parallel_batch(client, model_id, batches, requests_at_once, max_tokens, tracker)

    result = {
        "model": model_id,
        "max_tokens": max_tokens,
        "method": "mixed_parallel_batch",
        "timestamp": time.time(),
        "total_prompts": total_prompts,
        "total_tokens": tracker.total_tokens,
        "total_time": tracker.total_time,
        "average_tokens_per_second": calculate_tokens_per_second(tracker.total_tokens, tracker.total_time),
        "batch_results": tracker.batch_results,
    }

    def print_summary(data):
        print(f"Avg Tokens/Second: {data['average_tokens_per_second']:.2f}")
        print(f"Completions per request: {completions_per_request}")
        print(f"Requests at once: {requests_at_once}")

    _save_benchmark_results_with_summary(output_path, result, "SUMMARY (MIXED PARALLEL + BATCH)", print_summary)
