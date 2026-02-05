import argparse
import asyncio
import time
import json
from .base import BaseBenchmark
from .utils import BenchmarkTracker, get_prompts_and_batches, create_benchmark_result, check_finish_reasons

class ParallelBenchmark(BaseBenchmark):
    NAME = "benchmark"
    HELP = "Benchmark completion speed with different batch sizes (parallel requests)"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--batch-sizes",
            type=str,
            default="1,2,4,8",
            help="Comma-separated list of batch sizes to test",
        )

    async def run(self):
        batch_sizes = [int(x.strip()) for x in self.args.batch_sizes.split(",")]
        dataset_len = len(self.dataset.data)
        # Test 4 full batches per batch size
        num_batches_to_test = 4
        
        print(f"Benchmarking model (PARALLEL): {self.args.model}")
        print(f"Base dataset size: {dataset_len} prompts")
        print(f"Testing {num_batches_to_test} batches per batch size")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Testing batch sizes: {batch_sizes}\n")

        results = []
        all_sample_outputs = []  # Store sample outputs for verification
        
        for batch_size in batch_sizes:
            # Calculate prompts needed for this batch size
            num_prompts = batch_size * num_batches_to_test
            print(f"Testing batch size: {batch_size}")
            prompts, batches = get_prompts_and_batches(self.dataset, num_prompts, batch_size)
            tracker = BenchmarkTracker()
            
            for batch_idx, batch in enumerate(batches):
                start_time = time.time()
                tasks = [self.client.get_completion(self.args.model, prompt, self.args.max_tokens) for prompt in batch]
                responses = await asyncio.gather(*tasks)
                batch_time = time.time() - start_time

                check_finish_reasons(responses, batch_idx)
                batch_tokens = sum(resp["usage"]["completion_tokens"] for resp in responses)
                
                # SANITY CHECK: Verify token counts
                actual_token_lengths = []
                actual_text_lengths = []
                for resp in responses:
                    # Chat completions use 'message' with 'content', not 'text'
                    text = resp["choices"][0]["message"]["content"] or ""
                    actual_text_lengths.append(len(text))
                    actual_token_lengths.append(len(text.split()))
                
                total_actual_words = sum(actual_token_lengths)
                total_actual_chars = sum(actual_text_lengths)
                
                # Store first batch sample for each batch size for inspection
                if batch_idx == 0:
                    sample_data = {
                        "batch_size": batch_size,
                        "num_prompts": len(batch),
                        "reported_total_tokens": batch_tokens,
                        "actual_word_count": total_actual_words,
                        "actual_char_count": total_actual_chars,
                        "sample_outputs": [
                            {
                                "prompt": batch[i] if i < len(batch) else None,
                                "text": (responses[i]["choices"][0]["message"]["content"] or "")[:200] + ("..." if len(responses[i]["choices"][0]["message"]["content"] or "") > 200 else ""),
                                "full_length": len(responses[i]["choices"][0]["message"]["content"] or ""),
                                "word_count": len((responses[i]["choices"][0]["message"]["content"] or "").split()),
                                "reported_tokens": responses[i]["usage"]["completion_tokens"],
                            }
                            for i in range(min(3, len(responses)))  # First 3 samples
                        ]
                    }
                    all_sample_outputs.append(sample_data)
                
                batch_result = tracker.add_batch_result(batch_idx, len(batch), batch_tokens, batch_time)
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {batch_result['tokens_per_second']:.2f} tok/s"
                )
                print(f"    SANITY CHECK: {len(responses)} responses, ~{total_actual_words} words, {total_actual_chars} chars generated")
                
                # Warn if token count seems suspicious
                if total_actual_chars < batch_tokens * 0.5:  # Less than 0.5 chars per token is suspicious
                    print(f"    ⚠️  WARNING: Server reports {batch_tokens} tokens but only {total_actual_chars} chars generated!")

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
            "timestamp": time.time(),
            "results": results,
            "sample_outputs": all_sample_outputs,  # Include verification data
        }

        # Save sample outputs to separate file for inspection
        samples_filename = self.args.output.replace(".json", "_samples.json")
        with open(samples_filename, "w") as f:
            json.dump({"samples": all_sample_outputs}, f, indent=2)
        print(f"\nSaved sample outputs for verification to {samples_filename}")

        def summary_formatter(data):
            print("Batch Size | Avg Tokens/Second")
            print("-" * 35)
            for result in data["results"]:
                print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")
            
            # Add sanity check summary
            if "sample_outputs" in data:
                print("\n=== SANITY CHECK SUMMARY ===")
                print("Batch Size | Reported Tokens | Actual Words | Actual Chars | Ratio (Reported/Words)")
                print("-" * 95)
                for sample in data["sample_outputs"]:
                    ratio = sample["reported_total_tokens"] / sample["actual_word_count"] if sample["actual_word_count"] > 0 else 0
                    print(f"{sample['batch_size']:10} | {sample['reported_total_tokens']:15} | {sample['actual_word_count']:12} | {sample['actual_char_count']:12} | {ratio:>6.2f}x")

        self.save_results(output_data, "SUMMARY", summary_formatter)
