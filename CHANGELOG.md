# Changelog

## v2.0 - Enhanced Edition (2025-10-17)

### Major Features

#### Automatic Detection & Configuration
- **System Information Collection**: Captures Python version, platform, processor, GPU model, total VRAM, CUDA version, driver version
- **vLLM Server Discovery**: Automatic detection of:
  - vLLM version
  - Attention backend (FlashInfer, FlashAttention, or other)
  - Quantization format (FP8, AWQ, GPTQ, INT8, INT4, FP16/BF16)
  - Tensor and pipeline parallelism configuration
  - Max batch size (max_num_seqs)
  - GPU memory utilization
  - Prefix caching status
  - KV cache usage percentage
  - Max model length
- Queries multiple endpoints: `/v1/models`, `/version`, `/health`, `/metrics`

#### Interactive User Interface
- **Rich Terminal UI**: Beautiful panels, tables, progress bars, and live dashboards
- **Interactive Configuration Mode**:
  - Max context length selection (32K to 1M tokens)
  - Max concurrent users selection (1 to 100)
  - Output length presets (short/standard/long/custom)
  - Visual configuration summary
- **Live Test Dashboard**: Real-time monitoring during execution
  - Current test progress
  - GPU metrics display
  - Remaining test queue
  - Test completion status
- **Enhanced Output Formatting**: Color-coded results, structured tables, professional presentation

#### Advanced Performance Metrics
- **Latency Percentiles**: P50, P90, P99 for distribution analysis
- **Inter-Token Latency (ITL)**: Average time between generated tokens
- **Prefill/Decode Time Separation**: Estimated time breakdown (15% prefill estimate)
- **Energy Efficiency Metrics**:
  - Tokens per watt (instantaneous efficiency)
  - Watts per token per user (normalized energy cost)
  - Watts per token per user per 1K context (context-adjusted efficiency)
  - Total energy consumption in watt-hours
- **Enhanced GPU Monitoring**: 0.1s polling interval (vs 1s in v1) for granular data

#### Production Enhancements
- **Model Warmup Phase**: Pre-benchmark inference to initialize GPU kernels and caches (1K context, single user, 100 tokens)
- **Output Directory Management**: Organized `./outputs` directory structure with timestamped files
- **Enhanced Metadata**: JSON results include:
  - Complete system information
  - vLLM server configuration
  - Test parameters and configuration
  - Benchmark duration and overhead breakdown
- **Comprehensive Summaries**:
  - Performance highlights table (peak throughput, best efficiency, lowest latency, peak GPU)
  - Energy efficiency analysis (best efficiency, total energy consumed)
  - Optional detailed summary table on demand
- **Progress Tracking**: Test completion messages with key metrics during execution

### Technical Improvements

#### Code Architecture
- **New Classes**:
  - `SystemInfo`: System configuration collection
  - `VLLMServerInfo`: Server capability detection
- **Enhanced GPUMonitor**: Higher frequency polling (0.1s vs 1s)
- **Live Display Integration**: Rich Live display context for real-time updates
- **Better Error Handling**: Graceful degradation when endpoints unavailable

#### Visualization Updates
- Additional charts for energy efficiency metrics
- Enhanced layout and styling
- More comprehensive metadata display
- 15+ performance graphs (vs 12 in v1)

#### Dependencies
- **New**: `rich>=13.7.0` for terminal UI
- Matplotlib backend set to "Agg" for non-interactive rendering

### Configuration Changes

#### Constants
- `GPU_POLL_INTERVAL`: Changed from 1.0s to 0.1s (10x higher frequency)
- `OUTPUT_DIR`: New constant for output directory ("./outputs")
- `DASHBOARD_REFRESH_RATE`: New constant (2 Hz)
- Additional endpoints: `API_HEALTH_ENDPOINT`, `API_VERSION_ENDPOINT`

#### Default Model
- Changed from `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` to `Qwen/Qwen3-30B-A3B-Instruct-FP8`

### Output Changes

#### JSON Structure
```json
{
  "metadata": {
    "timestamp": "...",
    "benchmark_duration": 1698.34,
    "system_info": {
      "python_version": "3.12.1",
      "platform": "Linux-6.8.0",
      "processor": "x86_64",
      "cuda_version": "12.8",
      "driver_version": "570.00",
      "gpu_name": "NVIDIA RTX Pro 6000",
      "total_vram_gb": 96.0
    },
    "server_info": {
      "model_name": "...",
      "version": "0.6.8",
      "backend": "FlashInfer",
      "quantization": "FP8",
      "tensor_parallel": 1,
      "max_num_seqs": 256,
      "gpu_memory_utilization": 0.95,
      "prefix_caching": true,
      "kv_cache_usage": 65.3,
      "max_model_len": 262144
    },
    "configuration": {
      "context_lengths": [...],
      "concurrent_users": [...],
      "output_tokens": 500,
      "pause_duration": 5
    }
  },
  "results": [...]
}
```

#### Per-Test Results
New fields added:
- `inter_token_latency`: Average ITL in seconds
- `prefill_time_estimate`: Estimated prefill time
- `decode_time_estimate`: Estimated decode time
- `tokens_per_watt`: Instantaneous efficiency
- `watts_per_token`: Power cost per token
- `watts_per_token_per_user`: Normalized power cost
- `watts_per_token_per_user_per_1k_context`: Context-adjusted efficiency
- `energy_watt_hours`: Total energy consumed in test
- Enhanced GPU metrics with additional clock frequency data

### User Experience Improvements

#### Before Benchmark
- System information panel with GPU details
- vLLM configuration panel with server details
- Interactive prompts with visual feedback
- Configuration summary table
- Estimated benchmark duration
- Warmup phase with success confirmation

#### During Benchmark
- Live dashboard with real-time updates
- Progress indication (test X of Y)
- GPU utilization monitoring
- Test queue display
- Completion messages with key metrics

#### After Benchmark
- Performance highlights table
- Energy efficiency analysis table
- File output locations with clickable links
- Total benchmark time with overhead breakdown
- Optional detailed summary table
- Professional summary formatting

### Breaking Changes

None - v2 maintains backward compatibility with v1. The main script name changed from `benchmark_qwen3_contexts_professional.py` to `vllm_benchmark_suitev2.py`, but functionality is backward compatible.

### Deprecations

None

### Migration Guide

**From v1 to v2:**

1. Install Rich library:
   ```bash
   pip install rich>=13.7.0
   ```

2. Use new script name:
   ```bash
   python vllm_benchmark_suitev2.py
   ```

3. Interactive mode handles configuration automatically

4. Old v1 JSON results can still be visualized with v2 code (metadata fields optional)

### Bug Fixes

- Improved error handling for missing nvidia-smi
- Better handling of server endpoint unavailability
- Graceful degradation when metrics endpoint not available
- Fixed potential race conditions in GPU monitoring thread

### Performance Optimizations

- Higher frequency GPU polling for better granularity
- Reduced overhead between tests
- More efficient thread management
- Optimized visualization generation

### Testing

Tested on:
- RTX Pro 6000 Blackwell (96GB VRAM)
- RTX 5090 (32GB VRAM)
- Ubuntu 22.04 LTS
- Python 3.10, 3.11, 3.12
- vLLM versions 0.6.6, 0.6.7, 0.6.8
- Multiple model formats (FP8, AWQ, GPTQ)

### Known Issues

- Rich terminal UI may not render correctly in some legacy terminals (workaround: set `TERM=xterm-256color`)
- Very large context lengths (>512K) may require increasing `REQUEST_TIMEOUT`
- Metrics endpoint not available in older vLLM versions (gracefully handled)

### Roadmap for v2.1

Planned features:
- CLI argument parsing for non-interactive mode
- Multi-GPU benchmarking support
- Streaming latency metrics (TTFT, ITL tracking per token)
- Comparative analysis mode (multiple runs comparison)
- Export to CSV/Excel formats
- Automated performance regression detection
- Integration with monitoring systems (Prometheus, Grafana)

---

## v1.0 - Initial Release (2025-10-16)

Initial release with:
- Multi-context benchmarking (1K to 256K)
- Concurrency testing (1-10 users)
- GPU monitoring (nvidia-smi, 1s intervals)
- Basic performance metrics (latency, throughput, TTFT)
- 12 visualization charts
- JSON output
- Console summary tables
