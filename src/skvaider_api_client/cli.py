import argparse
import asyncio
import toml
from .config import Config
from .benchmarks import BENCHMARKS

async def _main():
    # load arguments
    parser = argparse.ArgumentParser(description="Skvaider API Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")

    # add subcommand
    subparser = parser.add_subparsers(dest="command")

    # Register benchmarks
    benchmark_map = {}
    for benchmark_cls in BENCHMARKS:
        b_parser = subparser.add_parser(benchmark_cls.NAME, help=benchmark_cls.HELP)
        benchmark_cls.add_arguments(b_parser)
        benchmark_map[benchmark_cls.NAME] = benchmark_cls

    args = parser.parse_args()
    
    # load config.toml
    try:
        with open(args.config, "r") as f:
            config_data = toml.load(f)
        config = Config(**config_data)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return

    if args.command in benchmark_map:
        benchmark_cls = benchmark_map[args.command]
        # Benchmark classes take config and args
        benchmark = benchmark_cls(config, args)
        await benchmark.run()
    else:
        parser.print_help()

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()
