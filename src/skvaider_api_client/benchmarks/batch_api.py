import argparse
import asyncio
import time
from .base import BaseBenchmark
from .utils import BenchmarkTracker, get_prompts_and_batches, create_benchmark_result, check_finish_reasons

class BatchApiBenchmark(BaseBenchmark):
    NAME = "benchmark-batch"
    HELP = "Benchmark completion speed using /completions batch API"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--batch-sizes",
            type=str,
            default="1,2,4,6",
            help="Comma-separated list of batch sizes to test",
        )

    async def run(self):
        batch_sizes = [int(x.strip()) for x in self.args.batch_sizes.split(",")]
        dataset_len = len(self.dataset.data)

        print(f"Benchmarking model (BATCH API): {self.args.model}")
        print(f"Base dataset size: {dataset_len} prompts")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Testing batch sizes: {batch_sizes}\n")

        results = []
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            prompts, batches = get_prompts_and_batches(self.dataset, dataset_len, batch_size)
            tracker = BenchmarkTracker()

            for batch_idx, batch in enumerate(batches):
                start_time = time.time()
                response = await self.client.get_batch_completion(self.args.model, batch, self.args.max_tokens)
                batch_time = time.time() - start_time

                check_finish_reasons([response], batch_idx)
                num_choices = len(response["choices"])
                tokens_per_choice = response["usage"]["completion_tokens"]
                batch_tokens = tokens_per_choice * num_choices
                batch_result = tracker.add_batch_result(batch_idx, len(batch), batch_tokens, batch_time)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {batch_result['tokens_per_second']:.2f} tok/s"
                )

            result = create_benchmark_result(
                batch_size, len(prompts), tracker.total_tokens, tracker.total_time, tracker.batch_results
            )
            results.append(result)
            print(
                f"  Overall: {tracker.total_tokens} tokens in {tracker.total_time:.2f}s = {result['average_tokens_per_second']:.2f} tok/s\n"
            )

        output_data = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "method": "batch_api",
            "timestamp": time.time(),
            "results": results,
        }

        def summary_formatter(data):
            print("Batch Size | Avg Tokens/Second")
            print("-" * 35)
            for result in data["results"]:
                print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")

        self.save_results(output_data, "SUMMARY (BATCH API)", summary_formatter)
