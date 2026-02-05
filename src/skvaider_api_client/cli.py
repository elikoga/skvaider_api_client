import argparse
import asyncio
import toml
from .benchmark import benchmark_batch_command, benchmark_command, benchmark_mixed_command
from .config import Config

async def _main():
    # load arguments
    parser = argparse.ArgumentParser(description="Skvaider API Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")

    # add subcommand
    subparser = parser.add_subparsers(dest="command")

    # benchmark subcommand
    b_parser = subparser.add_parser(
        "benchmark", help="Benchmark completion speed with different batch sizes (parallel requests)"
    )
    b_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    b_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    b_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    b_parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of batch sizes to test (prompts will be replicated if needed)",
    )
    b_parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Path to output benchmark JSON file"
    )

    # benchmark-batch subcommand
    bb_parser = subparser.add_parser("benchmark-batch", help="Benchmark completion speed using /completions batch API")
    bb_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    bb_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    bb_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    bb_parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,6",
        help="Comma-separated list of batch sizes to test (prompts will be replicated if needed)",
    )
    bb_parser.add_argument(
        "--output",
        type=str,
        default="benchmark_batch_results.json",
        help="Path to output benchmark JSON file",
    )

    # benchmark-mixed subcommand
    bm_parser = subparser.add_parser(
        "benchmark-mixed", help="Benchmark completion speed using mixed parallel and batch API approach"
    )
    bm_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    bm_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    bm_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    bm_parser.add_argument(
        "--completions-per-request", type=int, default=2, help="Number of completions to request per API call"
    )
    bm_parser.add_argument(
        "--requests-at-once", type=int, default=4, help="Number of parallel requests to make at once"
    )
    bm_parser.add_argument(
        "--output",
        type=str,
        default="benchmark_mixed_results.json",
        help="Path to output benchmark JSON file",
    )

    args = parser.parse_args()
    # load config.toml
    with open(args.config, "r") as f:
        config_data = toml.load(f)
    config = Config(**config_data)

    if args.command == "benchmark":
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        await benchmark_command(config, args.model, args.dataset, args.max_tokens, batch_sizes, args.output)
    elif args.command == "benchmark-batch":
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        await benchmark_batch_command(config, args.model, args.dataset, args.max_tokens, batch_sizes, args.output)
    elif args.command == "benchmark-mixed":
        await benchmark_mixed_command(
            config,
            args.model,
            args.dataset,
            args.max_tokens,
            args.completions_per_request,
            args.requests_at_once,
            args.output,
        )
    else:
        parser.print_help()

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()
