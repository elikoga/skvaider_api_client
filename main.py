# app goal: create snapshots of embeddings to allow us to preserve numerical stability of the flying circus AI model API over time
from calendar import c
import toml
from pydantic import BaseModel
import httpx
import asyncio
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Config(BaseModel):
    skvaider_token: str
    skvaider_url: str

class APIClient:
    token: str
    url: str
    timeout: httpx.Timeout
    headers: dict
    client: httpx.AsyncClient

    def __init__(self, config: Config):
        self.token = config.skvaider_token
        self.url = config.skvaider_url
        self.timeout = httpx.Timeout(60.0, read=60.0)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def list_models(self):
        response = await self.client.get(f"{self.url}/models")
        response.raise_for_status()
        return response.json()
    # {'object': 'list', 'data': [{'id': 'gpt-oss:20b', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}, {'id': 'gpt-oss:120b', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}, {'id': 'mistral-small3.2:latest', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}, {'id': 'bge-m3:567m', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}, {'id': 'embeddinggemma:300m', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}, {'id': 'nomic-embed-text:v1.5', 'object': 'model', 'created': 0, 'owned_by': 'skvaider'}]}
    # 1: gpt-oss:20b
    # 2: gpt-oss:120b
    # 3: mistral-small3.2:latest
    # 4: bge-m3:567m
    # 5: embeddinggemma:300m
    # 6: nomic-embed-text:v1.5

    # 4, 5, 6 are embedding models

    def is_embedding_model_domain_known(self, model_id: str) -> bool:
        model_id = model_id.lower()
        embedding_models = ["bge-m3:567m", "embeddinggemma:300m", "nomic-embed-text:v1.5"]
        return model_id in embedding_models

    async def get_embedding(self, model_id: str, input_text: str | list[str]) -> dict:
        if not self.is_embedding_model_domain_known(model_id):
            raise ValueError(f"Model {model_id} is not a known embedding model.")
        json_data = {
            "model": model_id,
            "input": input_text
        }
        response = await self.client.post(f"{self.url}/embeddings", json=json_data)
        response.raise_for_status()
        return response.json()

    async def get_completion(self, model_id: str, prompt: str, max_tokens: int = 100) -> dict:
        # Try chat/completions format first
        json_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False
        }
        response = await self.client.post(f"{self.url}/chat/completions", json=json_data)
        response.raise_for_status()
        return response.json()


    async def get_batch_completion(self, model_id: str, prompts: list[str], max_tokens: int = 100) -> dict:
        """Use /completions endpoint with prompt as a list for true batching"""
        json_data = {
            "model": model_id,
            "prompt": prompts,  # List of prompts for batching
            "max_tokens": max_tokens,
            "stream": False
        }
        response = await self.client.post(f"{self.url}/completions", json=json_data)
        response.raise_for_status()
        return response.json()

class Dataset:
    data: list[str]

    def __init__(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            self.data = [line.strip() for line in f.readlines() if line.strip()]  # remove empty lines

async def embeddings_command(config: Config, dataset_path: str, output_path: str):

    # Load dataset
    dataset = Dataset(dataset_path)

    client = APIClient(config)
    models = await client.list_models()
    print("Available models:", models)

    # For all models, for all data points, get embeddings
    final_embeddings:dict[str, dict[str, list[float]]] = {} # model_id -> {input_text: embedding}
    for model in models["data"]:
        model_id = model["id"]
        if not client.is_embedding_model_domain_known(model_id):
            continue
        print(f"Processing model: {model_id}")
        final_embeddings[model_id] = {}
        # send as batch
        embeddings_response = await client.get_embedding(model_id, dataset.data)
        for item in embeddings_response["data"]:
            input_text = dataset.data[item["index"]]
            embedding_vector = item["embedding"]
            final_embeddings[model_id][input_text] = embedding_vector
            print(f"  Obtained embedding for input index {item['index']}: {input_text[:30]}... (len={len(embedding_vector)})")
        print(f"Completed model: {model_id}, obtained {len(final_embeddings[model_id])} embeddings.")

    # Save final embeddings to output file
    import json
    with open(output_path, "w") as f:
        json.dump(final_embeddings, f, separators=(',', ':'))
    print(f"Saved embeddings to {output_path}")

async def compare_command(config: Config, embeddings_path1: str, embeddings_path2: str, output_path: str):
    import json
    from scipy.spatial.distance import cosine

    # Load embeddings files
    with open(embeddings_path1, "r") as f:
        embeddings1 = json.load(f)
    with open(embeddings_path2, "r") as f:
        embeddings2 = json.load(f)

    # Compare embeddings for each model present in both files
    common_models = set(embeddings1.keys()).intersection(set(embeddings2.keys()))
    comparison_results = []

    # calc magnitude of difference for each input
    for model_id in common_models:
        model_embeddings1 = embeddings1[model_id]
        model_embeddings2 = embeddings2[model_id]
        common_inputs = set(model_embeddings1.keys()).intersection(set(model_embeddings2.keys()))
        differences = []
        for input_text in common_inputs:
            vec1 = np.array(model_embeddings1[input_text]).reshape(1, -1)
            vec2 = np.array(model_embeddings2[input_text]).reshape(1, -1)
            cos_sim = cosine_similarity(vec1, vec2)[0][0]
            differences.append((input_text, cos_sim))
        avg_similarity = sum(sim for _, sim in differences) / len(differences) if differences else 0.0
        comparison_results.append((model_id, avg_similarity, differences))

    # Save comparison results to markdown file
    with open(output_path, "w") as f:
        f.write("# Embeddings Comparison Report\n\n")
        for model_id, avg_similarity, differences in comparison_results:
            f.write(f"## Model: {model_id}\n")
            f.write(f"- Average Cosine Similarity: {avg_similarity:.6f}\n")
            f.write("### Individual Input Similarities:\n")
            f.write("| Input Text (truncated) | Cosine Similarity |\n")
            f.write("|-----------------------|-------------------|\n")
            for input_text, cos_sim in differences:
                f.write(f"| {input_text[:30]}... | {cos_sim:.6f} |\n")
            f.write("\n")

    print(f"Saved comparison report to {output_path}")

async def benchmark_batch_command(config: Config, model_id: str, dataset_path: str, max_tokens: int, batch_sizes: list[int], output_path: str):
    """Benchmark using /completions endpoint with prompt list batching"""
    import json
    import time

    # Load dataset
    dataset = Dataset(dataset_path)
    base_prompts = dataset.data

    client = APIClient(config)
    
    print(f"Benchmarking model (BATCH API): {model_id}")
    print(f"Base dataset size: {len(base_prompts)} prompts")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Testing batch sizes: {batch_sizes}\n")

    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Replicate prompts if batch size is larger than dataset
        if batch_size > len(base_prompts):
            replications = (batch_size // len(base_prompts)) + 1
            prompts = (base_prompts * replications)[:batch_size]
            print(f"  Note: Dataset replicated to create {batch_size} prompts")
        else:
            prompts = base_prompts
        
        # Create batches of prompts
        batches = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batches.append(batch)
        
        total_tokens = 0
        total_time = 0.0
        batch_results = []

        for batch_idx, batch in enumerate(batches):
            start_time = time.time()
            
            # Use batch API - single request with list of prompts
            response = await client.get_batch_completion(model_id, batch, max_tokens)
            
            end_time = time.time()
            batch_time = end_time - start_time

            # print response for debugging
            # print(f"Response for batch {batch_idx}: {response}")
            assert all(choice["finish_reason"] == "length" for choice in response["choices"]), "Not all completions finished due to length"
            
            # Count tokens generated - the API returns one choice per prompt
            # The usage.completion_tokens is per choice, not total
            num_choices = len(response["choices"])
            tokens_per_choice = response["usage"]["completion_tokens"]
            batch_tokens = tokens_per_choice * num_choices
            
            total_tokens += batch_tokens
            total_time += batch_time
            
            tokens_per_second = batch_tokens / batch_time if batch_time > 0 else 0
            batch_results.append({
                "batch_idx": batch_idx,
                "batch_size": len(batch),
                "tokens": batch_tokens,
                "time": batch_time,
                "tokens_per_second": tokens_per_second
            })
            
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {tokens_per_second:.2f} tok/s")
        
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        result = {
            "batch_size": batch_size,
            "total_prompts": len(prompts),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "batch_results": batch_results
        }
        results.append(result)
        
        print(f"  Overall: {total_tokens} tokens in {total_time:.2f}s = {avg_tokens_per_second:.2f} tok/s\n")

    # Save results to JSON
    output_data = {
        "model": model_id,
        "max_tokens": max_tokens,
        "method": "batch_api",
        "timestamp": time.time(),
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved benchmark results to {output_path}")
    
    # Print summary
    print("\n=== SUMMARY (BATCH API) ===")
    print(f"Model: {model_id}")
    print(f"Batch Size | Avg Tokens/Second")
    print("-" * 35)
    for result in results:
        print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")

async def benchmark_command(config: Config, model_id: str, dataset_path: str, max_tokens: int, batch_sizes: list[int], output_path: str):
    import json
    import time

    # Load dataset
    dataset = Dataset(dataset_path)
    base_prompts = dataset.data

    client = APIClient(config)
    
    print(f"Benchmarking model (PARALLEL): {model_id}")
    print(f"Base dataset size: {len(base_prompts)} prompts")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Testing batch sizes: {batch_sizes}\n")

    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Replicate prompts if batch size is larger than dataset
        if batch_size > len(base_prompts):
            replications = (batch_size // len(base_prompts)) + 1
            prompts = (base_prompts * replications)[:batch_size]
            print(f"  Note: Dataset replicated to create {batch_size} prompts")
        else:
            prompts = base_prompts
        
        # Create batches of prompts
        batches = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batches.append(batch)
        
        total_tokens = 0
        total_time = 0.0
        batch_results = []

        for batch_idx, batch in enumerate(batches):
            start_time = time.time()
            
            # Run completions in parallel for this batch
            tasks = [client.get_completion(model_id, prompt, max_tokens) for prompt in batch]
            responses = await asyncio.gather(*tasks)

            
            end_time = time.time()
            batch_time = end_time - start_time
            
            assert all(all(choice["finish_reason"] == "length" for choice in resp["choices"]) for resp in responses), f"Not all completions finished due to length, batch_idx={batch_idx}, responses={responses}"

            # Count tokens generated
            batch_tokens = sum(resp["usage"]["completion_tokens"] for resp in responses)
            total_tokens += batch_tokens
            total_time += batch_time
            
            tokens_per_second = batch_tokens / batch_time if batch_time > 0 else 0
            batch_results.append({
                "batch_idx": batch_idx,
                "batch_size": len(batch),
                "tokens": batch_tokens,
                "time": batch_time,
                "tokens_per_second": tokens_per_second
            })
            
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {batch_tokens} tokens in {batch_time:.2f}s = {tokens_per_second:.2f} tok/s")
        
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        result = {
            "batch_size": batch_size,
            "total_prompts": len(prompts),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "batch_results": batch_results
        }
        results.append(result)
        
        print(f"  Overall: {total_tokens} tokens in {total_time:.2f}s = {avg_tokens_per_second:.2f} tok/s\n")

    # Save results to JSON
    output_data = {
        "model": model_id,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved benchmark results to {output_path}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Model: {model_id}")
    print(f"Batch Size | Avg Tokens/Second")
    print("-" * 35)
    for result in results:
        print(f"{result['batch_size']:10} | {result['average_tokens_per_second']:>16.2f}")

async def benchmark_mixed_command(config: Config, model_id: str, dataset_path: str, max_tokens: int, completions_per_request: int, requests_at_once: int, output_path: str):
    import json
    import time

    # Load dataset
    dataset = Dataset(dataset_path)
    base_prompts = dataset.data

    client = APIClient(config)
    
    print(f"Benchmarking model (MIXED): {model_id}")
    # print(f"Base dataset size: {len(base_prompts)} prompts")
    print(f"Max tokens per completion: {max_tokens}")
    print(f"Completions per request: {completions_per_request}")
    print(f"Requests at once: {requests_at_once}\n")

    total_prompts = completions_per_request * requests_at_once
    total_tokens = 0
    total_time = 0.0
    batch_results = []

    # extend dataset to have enough prompts
    if total_prompts > len(base_prompts):
        replications = (total_prompts // len(base_prompts)) + 1
        prompts = (base_prompts * replications)[:total_prompts]
        print(f"  Note: Dataset replicated to create {total_prompts} prompts")
    else:
        prompts = base_prompts[:total_prompts]

    batches = []
    for i in range(0, len(prompts), completions_per_request):
        batch = prompts[i:i + completions_per_request]
        batches.append(batch)

    for batch_idx in range(0, len(batches), requests_at_once):
        current_batches = batches[batch_idx:batch_idx + requests_at_once]
        start_time = time.time()
        
        # Create tasks for current set of requests
        tasks = [client.get_batch_completion(model_id, batch, max_tokens) for batch in current_batches]
        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        batch_time = end_time - start_time

        assert all(all(choice["finish_reason"] == "length" for choice in resp["choices"]) for resp in responses), f"Not all completions finished due to length, batch_idx={batch_idx}, responses={responses}"

        # Count tokens generated
        batch_tokens = sum(resp["usage"]["completion_tokens"] * len(resp["choices"]) for resp in responses)
        total_tokens += batch_tokens
        total_time += batch_time
        
        tokens_per_second = batch_tokens / batch_time if batch_time > 0 else 0
        batch_results.append({
            "batch_idx": batch_idx // requests_at_once,
            "requests": len(current_batches),
            "tokens": batch_tokens,
            "time": batch_time,
            "tokens_per_second": tokens_per_second
        })

        print(f"  Requests {batch_idx // requests_at_once + 1}/{(len(batches) + requests_at_once - 1) // requests_at_once}: {batch_tokens} tokens in {batch_time:.2f}s = {tokens_per_second:.2f} tok/s")

    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    result = {
        "model": model_id,
        "max_tokens": max_tokens,
        "method": "mixed_parallel_batch",
        "timestamp": time.time(),
        "total_prompts": total_prompts,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "average_tokens_per_second": avg_tokens_per_second,
        "batch_results": batch_results
    }
    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved benchmark results to {output_path}")

    # Print summary
    print("\n=== SUMMARY (MIXED PARALLEL + BATCH) ===")
    print(f"Model: {model_id}")
    print(f"Avg Tokens/Second: {avg_tokens_per_second:.2f}")
    # print params
    print(f"Completions per request: {completions_per_request}")
    print(f"Requests at once: {requests_at_once}")


async def main():
    # load arguments
    parser = argparse.ArgumentParser(description="Skvaider API Client")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")

    # add subcommand
    subparser = parser.add_subparsers(dest="command")

    # embeddings subcommand
    e_parser = subparser.add_parser("embeddings", help="Get embeddings for dataset")
    e_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file")
    e_parser.add_argument("--output", type=str, default="embeddings_output.json", help="Path to output embeddings JSON file")

    # compare subcommand
    c_parser = subparser.add_parser("compare", help="Compare embeddings between models")
    c_parser.add_argument("--embeddings1", type=str, required=True, help="Path to first embeddings JSON file")
    c_parser.add_argument("--embeddings2", type=str, required=True, help="Path to second embeddings JSON file")
    c_parser.add_argument("--output", type=str, default="embeddings_comparison.md", help="Path to output comparison markdown file")

    # benchmark subcommand
    b_parser = subparser.add_parser("benchmark", help="Benchmark completion speed with different batch sizes (parallel requests)")
    b_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    b_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    b_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    b_parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated list of batch sizes to test (prompts will be replicated if needed)")
    b_parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to output benchmark JSON file")

    # benchmark-batch subcommand
    bb_parser = subparser.add_parser("benchmark-batch", help="Benchmark completion speed using /completions batch API")
    bb_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    bb_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    bb_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    bb_parser.add_argument("--batch-sizes", type=str, default="1,2,4,6", help="Comma-separated list of batch sizes to test (prompts will be replicated if needed)")
    bb_parser.add_argument("--output", type=str, default="benchmark_batch_results.json", help="Path to output benchmark JSON file")

    # benchmark-mixed subcommand
    bm_parser = subparser.add_parser("benchmark-mixed", help="Benchmark completion speed using mixed parallel and batch API approach")
    bm_parser.add_argument("--model", type=str, default="gpt-oss:120b", help="Model ID to benchmark")
    bm_parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to dataset file with prompts")
    bm_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per completion")
    bm_parser.add_argument("--completions-per-request", type=int, default=2, help="Number of completions to request per API call")
    bm_parser.add_argument("--requests-at-once", type=int, default=4, help="Number of parallel requests to make at once")
    bm_parser.add_argument("--output", type=str, default="benchmark_mixed_results.json", help="Path to output benchmark JSON file")

    args = parser.parse_args()
    # load config.toml
    with open(args.config, "r") as f:
        config_data = toml.load(f)
    config = Config(**config_data)

    # if command is embeddings
    if args.command == "embeddings":
        await embeddings_command(config, args.dataset, args.output)
    elif args.command == "compare":
        await compare_command(config, args.embeddings1, args.embeddings2, args.output)
    elif args.command == "benchmark":
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        await benchmark_command(config, args.model, args.dataset, args.max_tokens, batch_sizes, args.output)
    elif args.command == "benchmark-batch":
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        await benchmark_batch_command(config, args.model, args.dataset, args.max_tokens, batch_sizes, args.output)
    elif args.command == "benchmark-mixed":
        await benchmark_mixed_command(config, args.model, args.dataset, args.max_tokens, args.completions_per_request, args.requests_at_once, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
