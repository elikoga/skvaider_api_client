import argparse
from abc import ABC, abstractmethod
from ..config import Config
from ..client import APIClient
from ..dataset import Dataset
from .utils import save_benchmark_results

class BaseBenchmark(ABC):
    NAME: str
    HELP: str = "Benchmark command"

    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.client = APIClient(config)
        # Initialize dataset only if the argument exists
        if hasattr(args, 'dataset'):
            self.dataset = Dataset(args.dataset)
        else:
            self.dataset = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Register arguments for this benchmark command"""
        parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
        parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
        parser.add_argument("--output", type=str, default=f"benchmark_{cls.NAME}_results.json", help="Path to output benchmark JSON file")

    @abstractmethod
    async def run(self):
        """Execute the benchmark logic"""
        pass

    def save_results(self, output_data: dict, summary_header: str, summary_formatter):
        """Save results and print summary"""
        save_benchmark_results(self.args.output, output_data)
        print(f"\n=== {summary_header} ===")
        print(f"Model: {output_data['model']}")
        summary_formatter(output_data)
