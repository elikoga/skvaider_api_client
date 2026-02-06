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
                content_token_lengths = []
                content_char_lengths = []
                reasoning_token_lengths = []
                reasoning_char_lengths = []
                
                for idx, resp in enumerate(responses):
                    # Chat completions use 'message' with 'content', not 'text'
                    message = resp["choices"][0]["message"]
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning_content") or ""
                    
                    content_char_lengths.append(len(content))
                    content_token_lengths.append(len(content.split()) if content else 0)
                    reasoning_char_lengths.append(len(reasoning))
                    reasoning_token_lengths.append(len(reasoning.split()) if reasoning else 0)
                
                total_content_words = sum(content_token_lengths)
                total_content_chars = sum(content_char_lengths)
                total_reasoning_words = sum(reasoning_token_lengths)
                total_reasoning_chars = sum(reasoning_char_lengths)
                total_all_words = total_content_words + total_reasoning_words
                total_all_chars = total_content_chars + total_reasoning_chars
                
                # Store first batch sample for each batch size for inspection
                if batch_idx == 0:
                    sample_data = {
                        "batch_size": batch_size,
                        "num_prompts": len(batch),
                        "reported_total_tokens": batch_tokens,
                        "content_word_count": total_content_words,
                        "content_char_count": total_content_chars,
                        "reasoning_word_count": total_reasoning_words,
                        "reasoning_char_count": total_reasoning_chars,
                        "total_word_count": total_all_words,
                        "total_char_count": total_all_chars,
                        "sample_outputs": [
                            {
                                "prompt": batch[i] if i < len(batch) else None,
                                "content": (responses[i]["choices"][0]["message"].get("content") or "")[:200] + ("..." if len(responses[i]["choices"][0]["message"].get("content") or "") > 200 else ""),
                                "reasoning": (responses[i]["choices"][0]["message"].get("reasoning_content") or "")[:200] + ("..." if len(responses[i]["choices"][0]["message"].get("reasoning_content") or "") > 200 else ""),
                                "content_length": len(responses[i]["choices"][0]["message"].get("content") or ""),
                                "reasoning_length": len(responses[i]["choices"][0]["message"].get("reasoning_content") or ""),
                                "content_word_count": len((responses[i]["choices"][0]["message"].get("content") or "").split()),
                                "reasoning_word_count": len((responses[i]["choices"][0]["message"].get("reasoning_content") or "").split()),
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
                
                # Build sanity check message
                if total_reasoning_words > 0:
                    print(f"    SANITY CHECK: {len(responses)} responses")
                    print(f"      Content: ~{total_content_words} words, {total_content_chars} chars")
                    print(f"      Reasoning: ~{total_reasoning_words} words, {total_reasoning_chars} chars")
                    print(f"      Total: ~{total_all_words} words, {total_all_chars} chars")
                else:
                    print(f"    SANITY CHECK: {len(responses)} responses, ~{total_content_words} words, {total_content_chars} chars")
                
                # Warn if total generated content doesn't match token count reasonably
                # Average token is ~4 chars, so check if we're within reasonable range
                expected_min_chars = batch_tokens * 2  # At least 2 chars per token
                if total_all_chars < expected_min_chars:
                    print(f"    ⚠️  WARNING: Server reports {batch_tokens} tokens but only {total_all_chars} total chars generated!")

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
                # Check if any samples have reasoning
                has_reasoning = any(sample.get("reasoning_word_count", 0) > 0 for sample in data["sample_outputs"])
                
                if has_reasoning:
                    print("Batch Size | Reported Tokens | Content Words | Reasoning Words | Total Words | Total Chars")
                    print("-" * 100)
                    for sample in data["sample_outputs"]:
                        print(f"{sample['batch_size']:10} | {sample['reported_total_tokens']:15} | {sample.get('content_word_count', 0):13} | {sample.get('reasoning_word_count', 0):15} | {sample.get('total_word_count', 0):11} | {sample.get('total_char_count', 0):11}")
                else:
                    print("Batch Size | Reported Tokens | Actual Words | Actual Chars | Ratio (Reported/Words)")
                    print("-" * 95)
                    for sample in data["sample_outputs"]:
                        words = sample.get("total_word_count") or sample.get("actual_word_count", 0)
                        chars = sample.get("total_char_count") or sample.get("actual_char_count", 0)
                        ratio = sample["reported_total_tokens"] / words if words > 0 else 0
                        print(f"{sample['batch_size']:10} | {sample['reported_total_tokens']:15} | {words:12} | {chars:12} | {ratio:>6.2f}x")

        self.save_results(output_data, "SUMMARY", summary_formatter)
