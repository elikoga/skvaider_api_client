# Skvaider API Client

benchmarking tool for testing throughput and performance of language model APIs

## Configuration

Create a `config.toml` file with your API credentials:

```toml
skvaider_token = "your-api-token"
skvaider_url = "https://your-api-endpoint.com"
```

You can specify a different config file with `--config path/to/config.toml`.

## Benchmark Modes

### 0. All Benchmarks (`benchmark-all`)

Runs all four benchmark modes with comprehensive settings. Includes a `--quick` flag for faster smoke testing.

**Example Usage:**

```bash
# Run all benchmarks with comprehensive settings
uv run skvaider-api-client benchmark-all --model openai/gpt-oss-120b

# Quick mode (fewer iterations, good for smoke testing)
uv run skvaider-api-client benchmark-all --model openai/gpt-oss-120b --quick
```

**What it tests:**
- `/chat/completions` parallel: 1, 4, 16, 32, 64, 128 batch sizes
- `/completions` batch API: 1, 4, 16, 32, 64, 128 batch sizes
- `/completions` mixed: (2,4,8 completions) × (4,8,16,32 requests)
- `/chat/completions` sustained: 4, 16, 32, 64, 128 requests in flight

### 1. Parallel Benchmark (`benchmark`) - `/chat/completions`

Tests throughput by sending multiple parallel chat completion requests in batches.

Sends requests using `/v1/chat/completions` endpoint in parallel and does asyncio.gather to wait for all responses. Measures total tokens generated, total time taken, and calculates tokens/second for each batch size.

**Default batch sizes:** 1, 4, 16, 32, 64, 128

**Example Usage:**

```bash
# Test with a single batch size
uv run skvaider-api-client benchmark --model openai/gpt-oss-120b --batch-sizes 4

# Test multiple batch sizes
uv run skvaider-api-client benchmark --model openai/gpt-oss-120b --batch-sizes 1,4,16,32,64,128

# Customize max tokens per completion
uv run skvaider-api-client benchmark --model openai/gpt-oss-120b --batch-sizes 4,8 --max-tokens 200
```

**Output:**
```
Benchmarking model (/chat/completions PARALLEL): openai/gpt-oss-120b
Testing batch sizes: [1, 4, 16, 32, 64, 128]

Testing batch size: 4
  Batch 1/4: 400 tokens in 2.73s = 146.69 tok/s
    SANITY CHECK: 4 responses
      Content: ~77 words, 483 chars
      Reasoning: ~320 words, 2011 chars
      Total: ~397 words, 2494 chars
  ...

=== SUMMARY ===
Batch Size | Avg Tokens/Second
-----------------------------------
         1 |            50.23
         2 |            98.45
         4 |           146.81
         8 |           215.33
```

### 2. Batch API Benchmark (`benchmark-batch`) - `/completions`

Tests the `/completions` batch API endpoint which processes multiple prompts in a single request.

**Default batch sizes:** 1, 4, 16, 32, 64, 128

**Example Usage:**

```bash
# Test single batch size
uv run skvaider-api-client benchmark-batch --model openai/gpt-oss-120b --batch-sizes 4

# Test multiple batch sizes
uv run skvaider-api-client benchmark-batch --model openai/gpt-oss-120b --batch-sizes 1,4,16,32,64,128

# Custom batch sizes
uv run skvaider-api-client benchmark-batch --model openai/gpt-oss-120b --batch-sizes 8 --num-batches 10
```

**Output:**
```
Benchmarking model (/completions BATCH API): openai/gpt-oss-120b
Testing batch sizes: [4, 16, 32, 64]

Testing batch size: 4
  Batch 1/4: 400 tokens in 3.22s = 124.21 tok/s
  ...

=== SUMMARY ===
Batch Size | Avg Tokens/Second
-----------------------------------
         4 |           124.21
         8 |           193.35
```

### 3. Mixed Benchmark (`benchmark-mixed`) - `/completions`

Combines parallel requests with batch API - multiple parallel requests, each containing a batch of prompts. Tests all combinations of parameters.

**Default settings:** 2,4,8 completions per request × 4,8,16,32 parallel requests

**Example Usage:**

```bash
# Single configuration
uv run skvaider-api-client benchmark-mixed --model openai/gpt-oss-120b \
  --completions-per-request 2 --requests-at-once 4

# Test all combinations
uv run skvaider-api-client benchmark-mixed --model openai/gpt-oss-120b \
  --completions-per-request 1,2,4 --requests-at-once 2,4

# Find optimal configuration
uv run skvaider-api-client benchmark-mixed --model openai/gpt-oss-120b \
  --completions-per-request 1,2,4,8 --requests-at-once 1,2,4,8,16
```

**Output:**
```
Benchmarking model (/completions MIXED): openai/gpt-oss-120b
Testing 12 combinations

=== SUMMARY ===
Completions/Req | Requests/Once | Effective Batch | Avg Tokens/Second
---------------------------------------------------------------------------
              1 |             2 |               2 |             91.53
              1 |             4 |               4 |            122.68
              2 |             2 |               4 |            124.21
              2 |             4 |               8 |            191.95
              4 |             2 |               8 |            193.35
              4 |             4 |              16 |            215.81

Best configuration:
  Completions per request: 4
  Requests at once: 4
  Effective batch size: 16
  Tokens/second: 215.81
```

### 4. Sustained Benchmark (`benchmark-sustained`) - `/chat/completions`

Measures steady-state throughput by maintaining a constant number of requests in flight. Includes warmup and cooldown phases for accurate measurement.

**Default requests in flight:** 4, 16, 32, 64, 128

**Example Usage:**

```bash
# Quick test with defaults (10 warmup, 50 measurement, 10 cooldown)
uv run skvaider-api-client benchmark-sustained --model openai/gpt-oss-120b \
  --requests-in-flight 4

# Test multiple concurrency levels
uv run skvaider-api-client benchmark-sustained --model openai/gpt-oss-120b \
  --requests-in-flight 4,16,32,64,128

# Custom measurement periods
uv run skvaider-api-client benchmark-sustained --model openai/gpt-oss-120b \
  --requests-in-flight 4,8 \
  --warmup-requests 20 \
  --measurement-requests 100 \
  --cooldown-requests 20

# Short test for development
uv run skvaider-api-client benchmark-sustained --model openai/gpt-oss-120b \
  --requests-in-flight 4 \
  --warmup-requests 5 \
  --measurement-requests 20 \
  --cooldown-requests 5
```

**Output:**
```
Benchmarking model (/chat/completions SUSTAINED): openai/gpt-oss-120b
Warmup requests: 10
Measurement requests: 50
Cooldown requests: 10

Testing with 4 requests in flight:
  Warmup complete, entering steady state measurement...
  Measurement complete, entering cooldown...
  Warmup: 500 tokens in 10.13s
  Steady State: 2000 tokens in 14.29s = 140.00 tok/s
  Cooldown: 500 tokens in 1.74s
  Total: 3000 tokens in 26.15s

=== SUMMARY ===
Requests in Flight | Sustained Tok/s | Peak Tok/s | Avg Latency (s)
---------------------------------------------------------------------------
                 2 |           50.09 |     200.00 |           3.289
                 4 |          140.00 |     400.00 |           3.370
                 8 |          131.33 |     800.00 |           5.111

Best sustained throughput:
  Requests in flight: 4
  Sustained tokens/second: 140.00
  Average request latency: 3.370s
```

## Common Options

All benchmarks support these options:

- `--model MODEL`: Model ID to benchmark (required)
- `--dataset DATASET`: Path to dataset file with prompts (default: `dataset.txt`)
- `--output OUTPUT`: Path to save JSON results (auto-generated if not specified)
- `--max-tokens MAX_TOKENS`: Maximum tokens per completion (default: 100)
- `--config CONFIG`: Path to config file (default: `config.toml`)

## Output Files

Each benchmark saves two files:

1. **Main results**: `benchmark_<type>_results.json` - Contains all metrics and configuration
2. **Sample outputs**: `benchmark_<type>_results_samples.json` - First batch samples for inspection

Example result structure:
```json
{
  "model": "openai/gpt-oss-120b",
  "max_tokens": 100,
  "timestamp": 1770370803.817327,
  "results": [
    {
      "batch_size": 4,
      "total_tokens": 1600,
      "total_time": 10.90,
      "average_tokens_per_second": 146.81
    }
  ]
}
```

## Dataset Format

Create a `dataset.txt` file with one prompt per line:

```
This is the first prompt to test.
Here is another prompt for benchmarking.
A third example prompt.
```

The tool will cycle through prompts if more are needed for testing.

## Reasoning Models

The tool automatically detects and properly counts reasoning tokens from models that output both `content` and `reasoning_content` fields. The sanity check displays both separately:

```
SANITY CHECK: 4 responses
  Content: ~77 words, 483 chars
  Reasoning: ~320 words, 2011 chars
  Total: ~397 words, 2494 chars
```
