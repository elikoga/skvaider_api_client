import argparse
import asyncio
import time
from .base import BaseBenchmark
from .parallel import ParallelBenchmark
from .batch_api import BatchApiBenchmark
from .mixed import MixedBenchmark
from .sustained import SustainedBenchmark


class AllBenchmark(BaseBenchmark):
    NAME = "benchmark-all"
    HELP = "Run all benchmarks with comprehensive settings"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Run a quick version with fewer iterations (good for smoke testing)",
        )

    async def run(self):
        print(f"Running ALL benchmarks for model: {self.args.model}")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Quick mode: {self.args.quick}")
        print("=" * 80)
        print()

        all_results = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "quick_mode": self.args.quick,
            "timestamp": time.time(),
            "benchmarks": {}
        }

        # 1. Parallel Benchmark
        print("\n" + "=" * 80)
        print("1. /chat/completions PARALLEL - Testing parallel request performance")
        print("=" * 80 + "\n")
        
        parallel_args = argparse.Namespace(
            model=self.args.model,
            dataset=self.args.dataset,
            output=self.args.output,
            max_tokens=self.args.max_tokens,
            batch_sizes="1,4,16,32,64,128" if not self.args.quick else "1,16,64",
        )
        parallel = ParallelBenchmark(self.config, parallel_args)
        await parallel.run()
        all_results["benchmarks"]["parallel"] = {
            "completed": True,
            "batch_sizes": parallel_args.batch_sizes,
        }

        # 2. Batch API Benchmark
        print("\n" + "=" * 80)
        print("2. /completions BATCH API - Testing batch API performance")
        print("=" * 80 + "\n")
        
        batch_args = argparse.Namespace(
            model=self.args.model,
            dataset=self.args.dataset,
            output=self.args.output,
            max_tokens=self.args.max_tokens,
            batch_sizes="1,4,16,32,64,128" if not self.args.quick else "1,16,64",
        )
        batch = BatchApiBenchmark(self.config, batch_args)
        await batch.run()
        all_results["benchmarks"]["batch_api"] = {
            "completed": True,
            "batch_sizes": batch_args.batch_sizes,
        }

        # 3. Mixed Benchmark
        print("\n" + "=" * 80)
        print("3. /completions MIXED - Testing parallel + batch combinations")
        print("=" * 80 + "\n")
        
        if self.args.quick:
            completions = "4"
            requests = "4,16"
        else:
            completions = "2,4,8"
            requests = "4,8,16,32"
            
        mixed_args = argparse.Namespace(
            model=self.args.model,
            dataset=self.args.dataset,
            output=self.args.output,
            max_tokens=self.args.max_tokens,
            completions_per_request=completions,
            requests_at_once=requests,
        )
        mixed = MixedBenchmark(self.config, mixed_args)
        await mixed.run()
        all_results["benchmarks"]["mixed"] = {
            "completed": True,
            "completions_per_request": completions,
            "requests_at_once": requests,
        }

        # 4. Sustained Benchmark
        print("\n" + "=" * 80)
        print("4. /chat/completions SUSTAINED - Testing sustained throughput")
        print("=" * 80 + "\n")
        
        sustained_args = argparse.Namespace(
            model=self.args.model,
            dataset=self.args.dataset,
            output=self.args.output,
            max_tokens=self.args.max_tokens,
            requests_in_flight="4,16,32,64,128" if not self.args.quick else "4,32,128",
            warmup_requests=10 if not self.args.quick else 5,
            measurement_requests=50 if not self.args.quick else 20,
            cooldown_requests=10 if not self.args.quick else 5,
        )
        sustained = SustainedBenchmark(self.config, sustained_args)
        await sustained.run()
        all_results["benchmarks"]["sustained"] = {
            "completed": True,
            "requests_in_flight": sustained_args.requests_in_flight,
        }

        # Save comprehensive results
        output_file = self.args.output.replace("_results.json", "_all_results.json")
        import json
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("ALL BENCHMARKS COMPLETED")
        print("=" * 80)
        print(f"\nComprehensive results saved to: {output_file}")
        print("\nIndividual benchmark results saved to:")
        print(f"  - Parallel: benchmark_{ParallelBenchmark.NAME}_results.json")
        print(f"  - Batch API: benchmark_{BatchApiBenchmark.NAME}_results.json")
        print(f"  - Mixed: benchmark_{MixedBenchmark.NAME}_results.json")
        print(f"  - Sustained: benchmark_{SustainedBenchmark.NAME}_results.json")
        print()
