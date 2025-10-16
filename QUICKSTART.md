# Quick Start Guide

Get the vLLM Performance Benchmark Suite running in 5 minutes.

## Prerequisites Check

```bash
# Check Python version (requires 3.10+)
python3 --version

# Check NVIDIA driver
nvidia-smi

# Check if vLLM server is accessible
curl http://localhost:8000/v1/models
```

## Installation (3 steps)

### Step 1: Clone and Setup

```bash
git clone https://github.com/yourusername/vllm-benchmark-suite.git
cd vllm-benchmark-suite
```

### Step 2: Install Dependencies

**Using uv (recommended - fastest):**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Using pip:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Start vLLM Server (if not running)

```bash
# Example: Start Qwen3-Next-80B
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95
```

## Run Benchmark

```bash
python benchmark_qwen3_contexts_professional.py
```

Expected runtime: 30-45 minutes for full benchmark (40 test configurations)

## Understanding Output

### During Execution

```
[Progress: 1/40]
Testing: 1,000 token context | 1 concurrent users
===================================

Results:
  Total time:              12.45s
  Successful:              1/1
  
Latency Metrics:
  Average:                 12.45s
  
Throughput Metrics:
  Tokens/second:           40.2
  
GPU Metrics:
  Avg utilization:         85.3%
  Avg memory used:         94215 MB (92.0 GB)
```

### After Completion

Three files are generated:

1. **benchmark_results_TIMESTAMP.json**
   - Raw data for further analysis
   - Import into pandas/Excel for custom analysis

2. **benchmark_TIMESTAMP.png**
   - 12+ performance visualization charts
   - Publication-ready at 300 DPI

3. **Console Summary Tables**
   - Optimal configurations
   - Peak performance metrics
   - Context scaling analysis

## Quick Customization

### Test Fewer Contexts (faster testing)

Edit `benchmark_qwen3_contexts_professional.py`:

```python
# Minimal test (4 configs, ~5 minutes)
context_lengths = [1000, 32000, 128000, 256000]
concurrent_users = [1]

# Balanced test (16 configs, ~15 minutes)
context_lengths = [1000, 32000, 64000, 128000, 192000, 256000]
concurrent_users = [1, 5]
```

### Change Output Token Count

```python
# Default: 500 tokens
output_tokens = 500

# Short responses (faster)
output_tokens = 100

# Long responses (stress test)
output_tokens = 2000
```

### Adjust Server URL

```python
API_BASE_URL = "http://localhost:8000"  # default

# Remote server
API_BASE_URL = "http://192.168.1.100:8000"

# Different port
API_BASE_URL = "http://localhost:9000"
```

## Common Issues

### Issue: "Connection refused"
**Solution**: Start vLLM server first
```bash
curl http://localhost:8000/v1/models
```

### Issue: "ModuleNotFoundError: No module named 'requests'"
**Solution**: Activate virtual environment and install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "nvidia-smi not found"
**Solution**: Install NVIDIA drivers or add to PATH
```bash
export PATH=$PATH:/usr/bin
```

### Issue: All requests timeout
**Solution**: Increase timeout or reduce context length
```python
REQUEST_TIMEOUT = 1800  # 30 minutes for very long contexts
```

## Next Steps

1. **Analyze Results**: Open the generated PNG to review performance
2. **Optimize Configuration**: Adjust vLLM settings based on findings
3. **Compare Models**: Run benchmarks with different models
4. **Share Results**: Post to community forums or research papers

## Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Review [Troubleshooting](README.md#troubleshooting) section
- Open an issue on GitHub
- Join the vLLM Discord community

## Sample Commands

### Full benchmark (default)
```bash
python benchmark_qwen3_contexts_professional.py
```

### Quick test (1 user only)
```bash
# Edit script to set: concurrent_users = [1]
python benchmark_qwen3_contexts_professional.py
```

### Production simulation (high concurrency)
```bash
# Edit script to set: concurrent_users = [10, 20, 50]
python benchmark_qwen3_contexts_professional.py
```

### Monitor in real-time
```bash
# Terminal 1: Run benchmark
python benchmark_qwen3_contexts_professional.py

# Terminal 2: Watch GPU
watch -n 1 nvidia-smi
```

Enjoy benchmarking!
