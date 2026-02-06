import argparse
import asyncio
import time
import json
from itertools import product
from .base import BaseBenchmark
from .utils import BenchmarkTracker, get_prompts_and_batches, check_finish_reasons, calculate_tokens_per_second

class MixedBenchmark(BaseBenchmark):
    NAME = "benchmark-mixed"
    HELP = "Benchmark /completions with mixed parallel requests + batch API"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--completions-per-request", type=str, default="2,4,8", help="Comma-separated list of completions per request to test"
        )
        parser.add_argument(
            "--requests-at-once", type=str, default="4,8,16,32", help="Comma-separated list of parallel requests to test"
        )

    async def run(self):
        # Parse comma-separated values
        completions_per_request_list = [int(x.strip()) for x in self.args.completions_per_request.split(",")]
        requests_at_once_list = [int(x.strip()) for x in self.args.requests_at_once.split(",")]
        
        # Generate all combinations
        combinations = list(product(completions_per_request_list, requests_at_once_list))
        
        print(f"Benchmarking model (/completions MIXED): {self.args.model}")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Testing {len(combinations)} combinations:")
        print(f"  Completions per request: {completions_per_request_list}")
        print(f"  Requests at once: {requests_at_once_list}\n")

        all_results = []
        
        for completions_per_request, requests_at_once in combinations:
            print(f"Testing completions_per_request={completions_per_request}, requests_at_once={requests_at_once}")
            
            total_prompts = completions_per_request * requests_at_once
            prompts, batches = get_prompts_and_batches(self.dataset, total_prompts, completions_per_request)
            tracker = BenchmarkTracker()
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
                "completions_per_request": completions_per_request,
                "requests_at_once": requests_at_once,
                "effective_batch_size": completions_per_request * requests_at_once,
                "total_prompts": total_prompts,
                "total_tokens": tracker.total_tokens,
                "total_time": tracker.total_time,
                "average_tokens_per_second": calculate_tokens_per_second(tracker.total_tokens, tracker.total_time),
                "batch_results": tracker.batch_results,
            }
            all_results.append(result)
            print(f"  Overall: {tracker.total_tokens} tokens in {tracker.total_time:.2f}s = {result['average_tokens_per_second']:.2f} tok/s\n")

        output_data = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "method": "mixed_parallel_batch",
            "timestamp": time.time(),
            "results": all_results,
        }

        def summary_formatter(data):
            results = data["results"]
            
            # Sort results by completions_per_request, then requests_at_once
            results = sorted(results, key=lambda x: (x["completions_per_request"], x["requests_at_once"]))
            
            print("Completions/Req | Requests/Once | Effective Batch | Avg Tokens/Second")
            print("-" * 75)
            for result in results:
                print(f"{result['completions_per_request']:15} | {result['requests_at_once']:13} | {result['effective_batch_size']:15} | {result['average_tokens_per_second']:>17.2f}")
            
            # Find best configuration
            best_result = max(results, key=lambda x: x["average_tokens_per_second"])
            print(f"\nBest configuration:")
            print(f"  Completions per request: {best_result['completions_per_request']}")
            print(f"  Requests at once: {best_result['requests_at_once']}")
            print(f"  Effective batch size: {best_result['effective_batch_size']}")
            print(f"  Tokens/second: {best_result['average_tokens_per_second']:.2f}")

        self.save_results(output_data, "SUMMARY (MIXED PARALLEL + BATCH)", summary_formatter)
