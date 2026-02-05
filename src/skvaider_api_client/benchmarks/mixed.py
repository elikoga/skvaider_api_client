import argparse
import asyncio
import time
from .base import BaseBenchmark
from .utils import BenchmarkTracker, get_prompts_and_batches, check_finish_reasons, calculate_tokens_per_second

class MixedBenchmark(BaseBenchmark):
    NAME = "benchmark-mixed"
    HELP = "Benchmark completion speed using mixed parallel and batch API approach"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--completions-per-request", type=int, default=2, help="Number of completions to request per API call"
        )
        parser.add_argument(
            "--requests-at-once", type=int, default=4, help="Number of parallel requests to make at once"
        )

    async def run(self):
        print(f"Benchmarking model (MIXED): {self.args.model}")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Completions per request: {self.args.completions_per_request}")
        print(f"Requests at once: {self.args.requests_at_once}\n")

        total_prompts = self.args.completions_per_request * self.args.requests_at_once
        prompts, batches = get_prompts_and_batches(self.dataset, total_prompts, self.args.completions_per_request)
        tracker = BenchmarkTracker()
        
        requests_at_once = self.args.requests_at_once
        max_tokens = self.args.max_tokens

        for batch_idx in range(0, len(batches), requests_at_once):
            current_batches = batches[batch_idx : batch_idx + requests_at_once]
            start_time = time.time()
            tasks = [self.client.get_batch_completion(self.args.model, batch, max_tokens) for batch in current_batches]
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

        result = {
            "model": self.args.model,
            "max_tokens": max_tokens,
            "method": "mixed_parallel_batch",
            "timestamp": time.time(),
            "total_prompts": total_prompts,
            "total_tokens": tracker.total_tokens,
            "total_time": tracker.total_time,
            "average_tokens_per_second": calculate_tokens_per_second(tracker.total_tokens, tracker.total_time),
            "batch_results": tracker.batch_results,
        }

        def summary_formatter(data):
            print(f"Avg Tokens/Second: {data['average_tokens_per_second']:.2f}")
            print(f"Completions per request: {self.args.completions_per_request}")
            print(f"Requests at once: {self.args.requests_at_once}")

        self.save_results(result, "SUMMARY (MIXED PARALLEL + BATCH)", summary_formatter)
