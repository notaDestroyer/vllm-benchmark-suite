# vLLM Performance Benchmark Suite v2.0

Comprehensive benchmarking tool for evaluating vLLM inference performance with automatic backend detection, advanced performance metrics, real-time monitoring, and interactive configuration.

## What's New in v2.0

### Automatic Detection & Configuration
- **System Information Collection**: Python version, platform, GPU model, VRAM, CUDA version, driver version
- **vLLM Server Discovery**: Automatic detection of vLLM version, attention backend (FlashInfer/FlashAttention), quantization format, tensor/pipeline parallelism, max batch size, prefix caching status, KV cache usage, and max context length
- **Backend Inference**: Automatically identifies FP8/AWQ/GPTQ/INT8/INT4/FP16 quantization from model names

### Enhanced User Interface
- **Rich Terminal UI**: Beautiful panels, tables, progress bars, and live dashboards powered by the Rich library
- **Interactive Configuration**: CLI prompts for max context length (up to 1M tokens), concurrent users (up to 100), and output length selection
- **Live Test Dashboard**: Real-time display of current test progress, GPU metrics, and remaining test queue
- **Comprehensive Summaries**: Post-benchmark analysis with performance highlights and energy efficiency metrics

### Advanced Performance Metrics
- **Latency Percentiles**: P50, P90, P99 for detailed latency distribution analysis
- **Inter-Token Latency (ITL)**: Average time between generated tokens
- **Prefill/Decode Separation**: Estimated time breakdown for prefill vs decode phases
- **Energy Efficiency**: Tokens per watt, watts per token per user, normalized efficiency metrics (per 1K context)
- **Energy Consumption**: Total watt-hours consumed during benchmark execution

### Improved Monitoring
- **High-Frequency GPU Polling**: 0.1s intervals (vs 1s in v1) for more granular data
- **Energy Metrics**: Real-time power consumption tracking and efficiency calculations
- **System Context**: Full hardware and software configuration captured in results

### Production Features
- **Model Warmup**: Pre-benchmark inference to initialize GPU kernels and caches
- **Output Management**: Organized results in `./outputs` directory with timestamped files
- **Enhanced Metadata**: Complete system info, server config, and test parameters in JSON output
- **Optional Detailed Reports**: Post-benchmark detailed summary tables on demand

## Architecture

The benchmark suite consists of:
- **SystemInfo**: Collects Python, platform, GPU, CUDA, and driver information
- **VLLMServerInfo**: Queries vLLM endpoints for configuration and capabilities
- **GPUMonitor**: High-frequency polling of nvidia-smi for real-time GPU metrics
- **Request Generator**: Concurrent HTTP request handling with thread pools
- **Metrics Collector**: Statistical analysis including percentiles, ITL, and energy metrics
- **Visualization Engine**: matplotlib/seaborn-based chart generation with 15+ performance graphs
- **Interactive CLI**: Rich-powered terminal interface for configuration and live monitoring

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX Pro 6000 Blackwell, RTX 5090)
- Minimum 8GB VRAM (16GB+ recommended for large models)
- Linux operating system (Ubuntu 22.04+ recommended)

### Software
- Python 3.10 or higher
- NVIDIA drivers with nvidia-smi available
- vLLM server running and accessible

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/notaDestroyer/vllm-benchmark-suite.git
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

**Requirements:**
```
requests>=2.31.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.26.0
rich>=13.7.0
```

## Usage

### Starting vLLM Server

Before running benchmarks, start your vLLM server:

```bash
vllm serve MODEL_NAME --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.95
```

Example configurations:

**Qwen3-30B with FlashInfer:**
```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

**GLM-4.5-Air with AWQ:**
```bash
vllm serve THUDM/GLM-4.5-Air-AWQ-4bit \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --quantization awq
```

### Running the Benchmark

Basic usage (interactive mode):
```bash
python vllm_benchmark_suitev2.py
```

The interactive CLI will guide you through:
1. System and server information display
2. Max context length selection (32K to 1M tokens)
3. Max concurrent users selection (1 to 100)
4. Output length selection (short/standard/long/custom)
5. Configuration confirmation
6. Automatic model warmup
7. Live benchmark execution with real-time monitoring
8. Comprehensive results and visualizations

### Output Files

The benchmark generates files in the `./outputs` directory:

1. **benchmark_MODEL_TIMESTAMP.json**: Complete performance data with metadata
   - System information (Python, GPU, CUDA, driver versions)
   - Server configuration (vLLM version, backend, quantization, parallelism)
   - Test parameters (context lengths, users, output tokens)
   - Detailed metrics for each test (latency, throughput, GPU, energy)

2. **benchmark_MODEL_TIMESTAMP.png**: Comprehensive visualization (300 DPI, 15+ charts)
   - Throughput vs context length
   - Latency distribution with percentiles
   - Throughput heatmap
   - Efficiency metrics (tokens/s per user)
   - Request throughput
   - TTFT estimates
   - Context scaling impact
   - Success rates
   - GPU utilization and VRAM usage
   - Power consumption and temperature
   - Clock frequencies
   - Energy efficiency metrics

## Performance Metrics

### Latency Metrics
- **Average Latency**: Mean request duration
- **Standard Deviation**: Latency variance across requests
- **Min/Max Latency**: Best and worst case performance
- **P50/P90/P99**: Latency percentiles for distribution analysis
- **TTFT (Time to First Token)**: Estimated prefill latency
- **ITL (Inter-Token Latency)**: Average time between tokens

### Throughput Metrics
- **Tokens/Second**: Overall generation throughput
- **Requests/Second**: Request processing rate
- **Tokens/Second/User**: Per-user efficiency metric

### GPU Metrics
- **Utilization**: GPU compute usage percentage
- **VRAM Usage**: Memory consumption in GB
- **Temperature**: GPU thermal state in Celsius
- **Power Draw**: Instantaneous power consumption in watts
- **Clock Frequencies**: GPU core and memory clocks in MHz

### Energy Efficiency Metrics
- **Tokens/Watt**: Throughput per watt of power consumed
- **Watts/Token/User**: Energy efficiency per concurrent user
- **Normalized Efficiency**: Watts per token per user per 1K context
- **Total Energy**: Watt-hours consumed during test execution

## Example Output

### Interactive Configuration
```
┌─ vLLM Performance Benchmark Suite ─┐
│ Enhanced Edition v2.0 - Interactive │
└─────────────────────────────────────┘

Initializing system detection...
Querying vLLM server...

╭─ System Information ──────────────╮
│ Python      3.12.1                │
│ Platform    Linux-6.8.0-49-generic│
│ GPU         NVIDIA RTX Pro 6000   │
│             (96GB)                │
│ Driver      570.00                │
│ CUDA        12.8                  │
╰───────────────────────────────────╯

╭─ vLLM Configuration ──────────────────╮
│ Model               Qwen3-30B-FP8     │
│ vLLM Version        0.6.8             │
│ Attention Backend   FlashInfer        │
│ Quantization        FP8               │
│ Max Context         262,144 tokens    │
│ GPU Mem Util        95.0%             │
│ Prefix Caching      Enabled           │
╰───────────────────────────────────────╯

••• Benchmark Configuration •••

Select maximum context length:
  [1] 32K
  [2] 64K
  [3] 128K
  [4] 256K
  [5] 512K
  [6] 1024K (1M)

Select max context: 3

Total tests: 28 (7 contexts × 4 user levels)
Estimated time: 25-35 minutes
```

### Live Test Dashboard
```
╭─ Current Test (12/28) ─────────────╮
│ Context    96K tokens              │
│ Users      5                       │
│ Elapsed    18.3s                   │
│ GPU        94%                     │
│ Status     RUNNING                 │
╰────────────────────────────────────╯

╭─ Queue (16 remaining) ─────────────╮
│ Remaining tests:                   │
│                                    │
│   1. 96K × 10 users                │
│   2. 128K × 1 users                │
│   3. 128K × 2 users                │
│   ...                              │
╰────────────────────────────────────╯
```

### Final Summary
```
════════════════════════════════════════
           Benchmark Complete
════════════════════════════════════════

╔════════════════════════════════════════╗
║       Performance Highlights           ║
╠════════════════════════════════════════╣
║ Peak Throughput │ 87.3 tok/s           ║
║                 │ 10 users @ 64K       ║
╠════════════════════════════════════════╣
║ Best Efficiency │ 43.6 tok/s/user      ║
║                 │ 2 users @ 32K        ║
╠════════════════════════════════════════╣
║ Lowest Latency  │ 11.47s               ║
║                 │ 1 users @ 10K        ║
╠════════════════════════════════════════╣
║ Peak GPU Util   │ 97.8%                ║
║                 │ 10 users @ 128K      ║
╚════════════════════════════════════════╝

╔════════════════════════════════════════╗
║     Energy Efficiency Analysis         ║
╠════════════════════════════════════════╣
║ Best Energy     │ 0.23 tok/W           ║
║   Efficiency    │ 5 users @ 64K        ║
╠════════════════════════════════════════╣
║ Total Energy    │ 15.34 Wh             ║
║   Used          │ All tests combined   ║
╚════════════════════════════════════════╝

Outputs:
  * Results: ./outputs/benchmark_qwen3_20251017_143052.json
  * Charts: ./outputs/benchmark_qwen3_20251017_143052.png

Total Benchmark Time: 28.3 minutes (1,698 seconds)
  Test execution: 1,620s | Overhead (pauses, etc): 78s
```

## Configuration

### Customizing API Settings

Modify constants at the top of the script:

```python
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 900  # seconds
GPU_POLL_INTERVAL = 0.1  # seconds (high frequency)
TEST_PAUSE_DURATION = 5  # seconds between tests
OUTPUT_DIR = "./outputs"  # output directory
```

### Interactive Configuration Options

The CLI provides pre-configured options and custom settings:

**Context Lengths:**
- 32K: 1K, 10K, 32K
- 64K: 1K, 10K, 32K, 64K
- 128K: 1K, 10K, 32K, 64K, 96K, 128K
- 256K: All above + 160K, 192K, 224K, 256K
- 512K: All above + 384K, 512K
- 1024K: All above + 768K, 1024K

**Concurrent Users:**
- Standard: 1, 2, 5, 10, 20, 50
- Custom: Any value

**Output Tokens:**
- Short summaries: 150 tokens
- Standard responses: 500 tokens
- Long reports: 1500 tokens
- Custom: Any value

## Use Cases

### Production Capacity Planning
Determine optimal configuration for expected workload:
- Context length requirements and scaling behavior
- Concurrent user capacity and batching efficiency
- Hardware utilization targets and bottlenecks
- Energy consumption and cost analysis

### Model Comparison
Benchmark different models or configurations:
- Quantization formats (FP8 vs AWQ vs GPTQ vs INT4)
- Model sizes (7B vs 30B vs 72B parameters)
- Attention backends (FlashInfer vs FlashAttention)
- MoE architectures vs dense models

### Infrastructure Optimization
Evaluate hardware and configuration changes:
- GPU memory allocation strategies
- Batch size and KV cache tuning
- Prefix caching impact
- Chunked prefill effectiveness
- Tensor parallelism scaling

### Energy Efficiency Analysis
Optimize for power consumption:
- Tokens per watt across configurations
- Power-limited vs compute-limited scenarios
- Efficiency vs throughput trade-offs
- Cost per token analysis

### Regression Testing
Track performance across vLLM versions:
- Version upgrade validation
- Performance regression detection
- Optimization verification
- Backend comparison

## Advanced Topics

### Server Detection Details

v2 automatically queries multiple vLLM endpoints:

**Model Information** (`/v1/models`):
- Model name and ID
- Creation timestamp

**Version Information** (`/version`):
- vLLM version string

**Metrics Endpoint** (`/metrics`, Prometheus format):
- KV cache usage percentage
- Number of running requests
- Various internal metrics

**Configuration Inference**:
- Quantization format from model name
- Backend detection from server response headers
- Tensor/pipeline parallelism from configuration

### Energy Efficiency Metrics Explained

**Tokens per Watt**: Instantaneous throughput efficiency
```
tokens_per_watt = tokens_per_second / avg_power_draw
```

**Watts per Token per User**: Normalized energy cost
```
watts_per_token_per_user = avg_power_draw / (tokens_per_second / concurrent_users)
```

**Normalized Efficiency**: Context-length adjusted metric
```
normalized_efficiency = watts_per_token_per_user / (context_length / 1000)
```

**Total Energy Consumption**: Watt-hours for test
```
energy_wh = (avg_power_draw * test_duration) / 3600
```

### GPU Monitoring Implementation

High-frequency polling (100ms) captures:
- GPU utilization percentage
- VRAM usage (used/total in MB)
- GPU temperature (Celsius)
- Power draw (watts)
- GPU clock frequency (MHz)
- Memory clock frequency (MHz)

Statistics computed:
- Mean, max, min for all metrics
- Per-test aggregation
- Full timeline data saved in results

## Troubleshooting

### Server Connection Issues

```
[ERROR] Failed to query model name: Connection refused
```

**Solution**: Ensure vLLM server is running:
```bash
curl http://localhost:8000/v1/models
# or
curl http://localhost:8000/health
```

### GPU Monitoring Failures

```
[WARNING] GPU monitoring error: nvidia-smi not found
```

**Solution**: Install NVIDIA drivers or add nvidia-smi to PATH:
```bash
nvidia-smi --version
which nvidia-smi
```

### Out of Memory Errors

```
[ERROR] All requests failed!
  Error: HTTP 500 (CUDA out of memory)
```

**Solutions**:
1. Reduce `--gpu-memory-utilization` (try 0.85 or 0.80)
2. Reduce `--max-model-len`
3. Lower concurrent users in benchmark
4. Enable `--enable-chunked-prefill` for large contexts

### Request Timeouts

```
[ERROR] Error: ('Connection aborted.', timeout())
```

**Solutions**:
1. Increase `REQUEST_TIMEOUT` in script (default: 900s)
2. Reduce output tokens for faster completion
3. Check if server is under heavy load

### Rich Library Display Issues

If terminal output is garbled:
```bash
# Set TERM environment variable
export TERM=xterm-256color

# Or disable live display by modifying script
# Comment out Live display sections if needed
```

## Performance Optimization Tips

### GPU Power Limits
Cap power draw for efficiency testing:
```bash
sudo nvidia-smi -pl 450  # Set 450W power limit
sudo nvidia-smi -pl 300  # Set 300W power limit
```

Reset to default:
```bash
sudo nvidia-smi -pl <default_power>  # Check nvidia-smi -q for default
```

### vLLM Configuration for Maximum Throughput
```bash
vllm serve MODEL \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --dtype auto
```

### vLLM Configuration for Energy Efficiency
```bash
vllm serve MODEL \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 128 \
  --enable-prefix-caching
```

### System Tuning
Disable CPU frequency scaling:
```bash
sudo cpupower frequency-set -g performance
```

Set GPU persistence mode:
```bash
sudo nvidia-smi -pm 1
```

## Comparing v1 and v2

**v1 Features:**
- Basic benchmarking
- GPU monitoring (1s intervals)
- 12 visualization charts
- JSON output
- Console summary

**v2 Enhancements:**
- Automatic system and server detection
- Interactive CLI with Rich UI
- Live test dashboard
- High-frequency GPU monitoring (0.1s intervals)
- Advanced metrics (P50/P90/P99, ITL, prefill/decode)
- Energy efficiency analysis
- Model warmup phase
- Organized output directory
- 15+ visualization charts
- Enhanced metadata and summaries
- Optional detailed reports

**Migration from v1**: v2 is backward compatible. Existing scripts work with v2, but interactive mode provides better UX.

## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add energy efficiency metrics'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

**Areas for Contribution:**
- Additional visualization types
- Multi-GPU benchmarking
- Streaming latency metrics
- Cost analysis features
- Cloud provider integration
- Automated performance regression detection

## License

MIT License - see LICENSE file for details

## Citation

If you use this benchmark suite in your research or testing:

```bibtex
@software{vllm_benchmark_suite_v2,
  title = {vLLM Performance Benchmark Suite v2.0},
  author = {amit},
  year = {2025},
  url = {https://github.com/notaDestroyer/vllm-benchmark-suite}
}
```

## Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for the inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention kernels
- Community contributors and testers

## Version History

### v2.0 (2025-01-17)
- Complete UI overhaul with Rich library
- Automatic system and server detection
- Interactive configuration mode
- Energy efficiency metrics
- Advanced latency analysis (P50/P90/P99, ITL)
- High-frequency GPU monitoring (0.1s)
- Model warmup phase
- Enhanced visualizations

### v1.0 (2024-12)
- Initial release
- Basic benchmarking functionality
- GPU monitoring
- Visualization suite
- JSON output

## Support

For vLLM-specific questions:
- vLLM Documentation: https://docs.vllm.ai/
- vLLM GitHub: https://github.com/vllm-project/vllm
