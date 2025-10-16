# vLLM Performance Benchmark Suite

A comprehensive benchmarking tool for evaluating vLLM inference performance across various context lengths and concurrency levels. Designed for production environments with real-time GPU monitoring and detailed performance analytics.

## Features

- **Multi-Context Benchmarking**: Test performance across context lengths from 1K to 256K tokens
- **Concurrency Testing**: Evaluate throughput under varying concurrent load (1-10 simultaneous users)
- **Real-Time GPU Monitoring**: Track utilization, memory, temperature, power draw, and clock frequencies
- **Comprehensive Metrics**: Latency, throughput, TTFT (Time to First Token), and efficiency metrics
- **Professional Visualizations**: Generate publication-quality charts with 12+ performance graphs
- **Dynamic Model Detection**: Automatically queries the vLLM server for model information
- **Production-Ready**: Type hints, comprehensive error handling, and logging

## Architecture

The benchmark suite consists of:
- **GPUMonitor**: Background thread polling nvidia-smi for real-time GPU metrics
- **Request Generator**: Concurrent HTTP request handling with thread pools
- **Metrics Collector**: Statistical analysis of latency, throughput, and resource utilization
- **Visualization Engine**: matplotlib/seaborn-based chart generation

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX Pro 6000 Blackwell)
- Minimum 8GB VRAM (16GB+ recommended for large models)
- Linux operating system (Ubuntu 22.04+ recommended)

### Software
- Python 3.10 or higher
- NVIDIA drivers with nvidia-smi available
- vLLM server running and accessible

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vllm-benchmark-suite.git
cd vllm-benchmark-suite
```

### 2. Create Virtual Environment

Using `uv` (recommended):
```bash
uv venv venv --python 3.12
source venv/bin/activate.fish  # for fish shell
# or
source venv/bin/activate  # for bash/zsh
```

Using standard Python:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting vLLM Server

Before running benchmarks, start your vLLM server:

```bash
vllm serve MODEL_NAME --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.95
```

Example with Qwen3-Next-80B:
```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95
```

### Running the Benchmark

Basic usage:
```bash
python benchmark_qwen3_contexts_professional.py
```

The script will automatically:
1. Detect the model name from the vLLM server
2. Run 40 test configurations (10 context lengths × 4 concurrency levels)
3. Generate real-time console output with progress tracking
4. Save results to JSON and PNG files

### Output Files

The benchmark generates three output files:

1. **benchmark_results_TIMESTAMP.json**: Raw performance data
2. **benchmark_TIMESTAMP.png**: Comprehensive visualization (300 DPI)
3. **Console output**: Detailed summary tables and optimal configurations

## Configuration

### Customizing Test Parameters

Edit the `main()` function in `benchmark_qwen3_contexts_professional.py`:

```python
# Context lengths to test (in tokens)
context_lengths = [
    1000, 10000, 32000, 64000, 96000,
    128000, 160000, 192000, 224000, 256000
]

# Concurrent user counts
concurrent_users = [1, 2, 5, 10]

# Output tokens per request
output_tokens = 500
```

### Adjusting API Settings

Modify constants at the top of the script:

```python
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 900  # seconds
GPU_POLL_INTERVAL = 1  # seconds
TEST_PAUSE_DURATION = 5  # seconds between tests
```

## Performance Metrics

### Latency Metrics
- **Average Latency**: Mean request duration
- **Standard Deviation**: Latency variance across requests
- **Min/Max Latency**: Best and worst case performance
- **TTFT (Time to First Token)**: Estimated prefill latency

### Throughput Metrics
- **Tokens/Second**: Overall generation throughput
- **Requests/Second**: Request processing rate
- **Tokens/Second/User**: Per-user efficiency metric

### GPU Metrics
- **Utilization**: GPU compute usage (%)
- **VRAM Usage**: Memory consumption (GB)
- **Temperature**: GPU thermal state (°C)
- **Power Draw**: Instantaneous power consumption (W)
- **Clock Frequencies**: GPU core and memory clocks (MHz)

## Example Output

### Console Summary
```
================================================================================
DETAILED PERFORMANCE SUMMARY
================================================================================

Context Length: 32,000 tokens (32K)
--------------------------------------------------------------------------------
Users    Latency(s)   Tok/s      Req/s      TTFT(ms)   GPU%     VRAM(GB)   
1        12.45        40.2       0.08       1245       85.3     92.1       
2        24.12        41.4       0.08       2412       89.7     92.3       
5        58.34        42.8       0.09       5834       94.2     92.8       
10       115.67       43.2       0.09       11567      97.1     93.2       

================================================================================
OPTIMAL CONFIGURATIONS
================================================================================

Maximum Throughput:
  43.2 tokens/s at 10 users with 32K context
  GPU: 97.1% util, 93.2GB VRAM, 68.5°C, 425.3W, 2520 MHz
```

### Visualization

The generated PNG includes:
- Throughput vs context length (line chart)
- Latency distribution (line chart)
- Throughput heatmap
- Efficiency metrics (tokens/s per user)
- Request throughput
- TTFT estimates
- Context scaling impact
- Success rates
- GPU utilization
- VRAM usage
- Power and temperature
- Clock frequencies

## Use Cases

### Production Capacity Planning
Determine optimal configuration for expected workload:
- Context length requirements
- Concurrent user capacity
- Hardware utilization targets

### Model Comparison
Benchmark different models or quantizations:
- FP8 vs AWQ vs GPTQ
- 7B vs 72B vs 80B parameter models
- Flash Attention vs FlashInfer backends

### Infrastructure Optimization
Evaluate hardware and configuration changes:
- GPU memory allocation strategies
- Batch size tuning
- KV cache optimization

### Regression Testing
Track performance across vLLM versions:
- Version upgrade validation
- Performance regression detection
- Optimization verification

## Troubleshooting

### Server Connection Issues

```
[ERROR] Failed to query model name: Connection refused
```

**Solution**: Ensure vLLM server is running on localhost:8000

```bash
curl http://localhost:8000/v1/models
```

### GPU Monitoring Failures

```
[WARNING] GPU monitoring error: nvidia-smi not found
```

**Solution**: Install NVIDIA drivers or add nvidia-smi to PATH

```bash
nvidia-smi --version
```

### Out of Memory Errors

```
[ERROR] All requests failed!
  Error: HTTP 500
```

**Solution**: Reduce `--gpu-memory-utilization` or `--max-model-len`

### Request Timeouts

```
[ERROR] Error: ('Connection aborted.', timeout())
```

**Solution**: Increase `REQUEST_TIMEOUT` for longer contexts

## Performance Optimization Tips

### GPU Power Limits
Cap power draw for efficiency testing:
```bash
sudo nvidia-smi -pl 450  # Set 450W power limit
```

### vLLM Configuration
Optimize server settings:
```bash
vllm serve MODEL \
  --enforce-eager \
  --disable-log-stats \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256
```

### System Tuning
Disable CPU frequency scaling:
```bash
sudo cpupower frequency-set -g performance
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new metric'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details


## Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for the inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention kernels

