import argparse
import asyncio
import time
from collections import deque
from .base import BaseBenchmark
from .utils import calculate_tokens_per_second


class SustainedBenchmark(BaseBenchmark):
    NAME = "benchmark-sustained"
    HELP = "Benchmark sustained throughput with constant requests in flight"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
        parser.add_argument(
            "--requests-in-flight",
            type=str,
            default="4",
            help="Comma-separated list of concurrent requests to maintain",
        )
        parser.add_argument(
            "--warmup-requests",
            type=int,
            default=10,
            help="Number of requests to complete before measuring (warmup)",
        )
        parser.add_argument(
            "--measurement-requests",
            type=int,
            default=50,
            help="Number of requests to measure during steady state",
        )
        parser.add_argument(
            "--cooldown-requests",
            type=int,
            default=10,
            help="Number of requests after measurement (cooldown)",
        )

    async def run(self):
        requests_in_flight_list = [int(x.strip()) for x in self.args.requests_in_flight.split(",")]
        
        print(f"Benchmarking model (SUSTAINED): {self.args.model}")
        print(f"Max tokens per completion: {self.args.max_tokens}")
        print(f"Warmup requests: {self.args.warmup_requests}")
        print(f"Measurement requests: {self.args.measurement_requests}")
        print(f"Cooldown requests: {self.args.cooldown_requests}")
        print(f"Testing request-in-flight values: {requests_in_flight_list}\n")

        all_results = []

        for requests_in_flight in requests_in_flight_list:
            print(f"Testing with {requests_in_flight} requests in flight:")
            
            result = await self._run_sustained_test(requests_in_flight)
            all_results.append(result)
            
            print(f"  Warmup: {result['warmup_tokens']} tokens in {result['warmup_time']:.2f}s")
            print(f"  Steady State: {result['measurement_tokens']} tokens in {result['measurement_time']:.2f}s = {result['sustained_tokens_per_second']:.2f} tok/s")
            print(f"  Cooldown: {result['cooldown_tokens']} tokens in {result['cooldown_time']:.2f}s")
            print(f"  Total: {result['total_tokens']} tokens in {result['total_time']:.2f}s\n")

        output_data = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "method": "sustained",
            "timestamp": time.time(),
            "warmup_requests": self.args.warmup_requests,
            "measurement_requests": self.args.measurement_requests,
            "cooldown_requests": self.args.cooldown_requests,
            "results": all_results,
        }

        def summary_formatter(data):
            results = data["results"]
            
            print("Requests in Flight | Sustained Tok/s | Peak Tok/s | Avg Latency (s)")
            print("-" * 75)
            for result in results:
                print(
                    f"{result['requests_in_flight']:18} | {result['sustained_tokens_per_second']:15.2f} | "
                    f"{result['peak_tokens_per_second']:10.2f} | {result['avg_request_latency']:15.3f}"
                )
            
            # Find best sustained throughput
            best_result = max(results, key=lambda x: x["sustained_tokens_per_second"])
            print(f"\nBest sustained throughput:")
            print(f"  Requests in flight: {best_result['requests_in_flight']}")
            print(f"  Sustained tokens/second: {best_result['sustained_tokens_per_second']:.2f}")
            print(f"  Average request latency: {best_result['avg_request_latency']:.3f}s")

        self.save_results(output_data, "SUMMARY (SUSTAINED)", summary_formatter)

    async def _run_sustained_test(self, requests_in_flight: int):
        """Run a sustained throughput test maintaining N requests in flight"""
        total_requests = self.args.warmup_requests + self.args.measurement_requests + self.args.cooldown_requests
        
        # Prepare prompts (cycle through dataset)
        prompts = []
        for i in range(total_requests):
            prompts.append(self.dataset.data[i % len(self.dataset.data)])
        
        # Track request completion info
        completed_requests = []
        request_latencies = []
        
        # Phase tracking
        warmup_end = self.args.warmup_requests
        measurement_end = warmup_end + self.args.measurement_requests
        
        # Metrics for each phase
        warmup_tokens = 0
        warmup_time = 0
        measurement_tokens = 0
        measurement_start_time = None
        measurement_end_time = None
        cooldown_tokens = 0
        cooldown_time = 0
        
        overall_start_time = time.time()
        
        # Queue of pending requests
        pending = deque(range(total_requests))
        active_tasks = {}
        next_request_id = 0
        completed_count = 0
        
        async def make_request(request_id: int, prompt: str):
            """Make a single request and return its metadata"""
            req_start = time.time()
            response = await self.client.get_completion(self.args.model, prompt, self.args.max_tokens)
            req_end = time.time()
            
            tokens = response["usage"]["completion_tokens"]
            latency = req_end - req_start
            
            return {
                "request_id": request_id,
                "tokens": tokens,
                "latency": latency,
                "completed_at": req_end,
                "started_at": req_start,
            }
        
        # Initial fill: launch requests_in_flight requests
        for _ in range(min(requests_in_flight, len(pending))):
            request_id = pending.popleft()
            task = asyncio.create_task(make_request(request_id, prompts[request_id]))
            active_tasks[task] = request_id
        
        # Process requests as they complete, maintaining requests_in_flight
        while active_tasks or pending:
            if not active_tasks:
                break
                
            # Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            
            # Process completed tasks
            for task in done:
                request_id = active_tasks.pop(task)
                result = await task
                
                completed_count += 1
                completed_requests.append(result)
                request_latencies.append(result["latency"])
                
                # Determine which phase this request belongs to
                if completed_count <= warmup_end:
                    # Warmup phase
                    warmup_tokens += result["tokens"]
                    if completed_count == warmup_end:
                        warmup_time = result["completed_at"] - overall_start_time
                        measurement_start_time = result["completed_at"]
                        print(f"  Warmup complete, entering steady state measurement...")
                
                elif completed_count <= measurement_end:
                    # Measurement phase
                    measurement_tokens += result["tokens"]
                    if completed_count == measurement_end:
                        measurement_end_time = result["completed_at"]
                        print(f"  Measurement complete, entering cooldown...")
                
                else:
                    # Cooldown phase
                    cooldown_tokens += result["tokens"]
                
                # Launch a new request if we have more to do
                if pending:
                    new_request_id = pending.popleft()
                    new_task = asyncio.create_task(make_request(new_request_id, prompts[new_request_id]))
                    active_tasks[new_task] = new_request_id
        
        overall_end_time = time.time()
        
        # Calculate cooldown time
        if measurement_end_time:
            cooldown_time = overall_end_time - measurement_end_time
        
        # Calculate measurement time
        if measurement_start_time and measurement_end_time:
            measurement_time = measurement_end_time - measurement_start_time
        else:
            measurement_time = 0
        
        # Calculate metrics
        total_time = overall_end_time - overall_start_time
        total_tokens = sum(r["tokens"] for r in completed_requests)
        
        sustained_tokens_per_second = calculate_tokens_per_second(measurement_tokens, measurement_time) if measurement_time > 0 else 0
        overall_tokens_per_second = calculate_tokens_per_second(total_tokens, total_time)
        
        # Calculate peak throughput (best 1-second window)
        peak_tokens_per_second = self._calculate_peak_throughput(completed_requests)
        
        # Average latency
        avg_latency = sum(request_latencies) / len(request_latencies) if request_latencies else 0
        
        return {
            "requests_in_flight": requests_in_flight,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "warmup_tokens": warmup_tokens,
            "warmup_time": warmup_time,
            "measurement_tokens": measurement_tokens,
            "measurement_time": measurement_time,
            "cooldown_tokens": cooldown_tokens,
            "cooldown_time": cooldown_time,
            "sustained_tokens_per_second": sustained_tokens_per_second,
            "overall_tokens_per_second": overall_tokens_per_second,
            "peak_tokens_per_second": peak_tokens_per_second,
            "avg_request_latency": avg_latency,
            "min_request_latency": min(request_latencies) if request_latencies else 0,
            "max_request_latency": max(request_latencies) if request_latencies else 0,
        }
    
    def _calculate_peak_throughput(self, completed_requests, window_size=1.0):
        """Calculate peak tokens/second in any window_size second window"""
        if not completed_requests:
            return 0
        
        # Sort by completion time
        sorted_requests = sorted(completed_requests, key=lambda x: x["completed_at"])
        
        max_tokens_per_second = 0
        
        # Sliding window
        for i in range(len(sorted_requests)):
            window_start = sorted_requests[i]["completed_at"]
            window_end = window_start + window_size
            
            # Count tokens in this window
            tokens_in_window = 0
            for req in sorted_requests[i:]:
                if req["completed_at"] <= window_end:
                    tokens_in_window += req["tokens"]
                else:
                    break
            
            tokens_per_second = tokens_in_window / window_size
            max_tokens_per_second = max(max_tokens_per_second, tokens_per_second)
        
        return max_tokens_per_second
