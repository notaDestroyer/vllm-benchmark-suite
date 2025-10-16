# Configuration Guide

Detailed configuration options for the vLLM Performance Benchmark Suite.

## Configuration File Structure

All configuration is done through constants and variables in `benchmark_qwen3_contexts_professional.py`.

## Core Configuration

### API Endpoints

```python
# Server configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
API_MODELS_ENDPOINT = f"{API_BASE_URL}/v1/models"
```

**Options:**
- `localhost:8000` - Default vLLM server
- `0.0.0.0:8000` - Server accessible from network
- Remote server IP for distributed testing

### Timeouts and Intervals

```python
REQUEST_TIMEOUT = 900  # seconds (15 minutes)
GPU_POLL_INTERVAL = 1  # seconds
TEST_PAUSE_DURATION = 5  # seconds between tests
```

**Recommendations:**
- `REQUEST_TIMEOUT`: Increase for very long contexts (>200K tokens)
  - 1K-32K contexts: 300s (5 min)
  - 64K-128K contexts: 900s (15 min)
  - 192K-256K contexts: 1800s (30 min)

- `GPU_POLL_INTERVAL`: Balance between granularity and overhead
  - Fine-grained monitoring: 0.5s
  - Standard monitoring: 1s
  - Coarse monitoring: 2s

- `TEST_PAUSE_DURATION`: GPU cooldown between tests
  - Minimal pause: 2s
  - Standard pause: 5s
  - Full cooldown: 10s

## Test Parameters

### Context Lengths

```python
context_lengths = [
    1000,     # 1K - Baseline
    10000,    # 10K - Short documents
    32000,    # 32K - Medium documents
    64000,    # 64K - Long documents
    96000,    # 96K - Very long documents
    128000,   # 128K - Standard max for many models
    160000,   # 160K - Extended context
    192000,   # 192K - Near maximum
    224000,   # 224K - Very long context
    256000    # 256K - Maximum supported
]
```

**Preset Configurations:**

**Minimal (4 tests, ~5 min):**
```python
context_lengths = [1000, 64000, 128000, 256000]
concurrent_users = [1]
```

**Quick (12 tests, ~15 min):**
```python
context_lengths = [1000, 32000, 64000, 128000, 192000, 256000]
concurrent_users = [1, 5]
```

**Standard (40 tests, ~45 min):**
```python
context_lengths = [1000, 10000, 32000, 64000, 96000, 
                   128000, 160000, 192000, 224000, 256000]
concurrent_users = [1, 2, 5, 10]
```

**Comprehensive (100 tests, ~2 hours):**
```python
context_lengths = [1000, 5000, 10000, 20000, 32000, 48000,
                   64000, 80000, 96000, 112000, 128000, 
                   160000, 192000, 224000, 256000]
concurrent_users = [1, 2, 5, 10, 20, 50]
```

### Concurrency Levels

```python
concurrent_users = [1, 2, 5, 10]
```

**Guidelines:**
- **1 user**: Baseline single-user performance
- **2-5 users**: Light concurrent load
- **10-20 users**: Moderate concurrent load
- **50+ users**: Heavy production load

**Memory Considerations:**
- Each concurrent user increases memory usage
- Monitor VRAM usage during testing
- Reduce concurrency if OOM errors occur

### Output Token Configuration

```python
output_tokens = 500
```

**Use Cases:**
- **100 tokens**: Short responses, chatbot replies
- **500 tokens**: Standard responses, summaries
- **1000 tokens**: Long-form content, detailed analysis
- **2000+ tokens**: Document generation, comprehensive reports

**Impact on Performance:**
- Higher output tokens = longer latency
- Throughput (tokens/s) typically increases with longer outputs
- Memory usage scales linearly with output length

## GPU Monitoring Configuration

### Polling Frequency

```python
class GPUMonitor:
    def __init__(self, poll_interval: float = GPU_POLL_INTERVAL):
        self.poll_interval = poll_interval
```

**Trade-offs:**
- **Faster polling (0.5s)**
  - Pros: More granular data, better peak detection
  - Cons: Higher CPU overhead, more data points

- **Slower polling (2s)**
  - Pros: Lower overhead, simpler analysis
  - Cons: May miss short spikes

### Metrics Collected

Default metrics from nvidia-smi:
```python
'--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem'
```

**Add Additional Metrics:**
```python
# Example: Add fan speed and power state
'--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem,fan.speed,pstate'
```

Available nvidia-smi metrics:
- `fan.speed` - Fan speed percentage
- `pstate` - Performance state (P0-P12)
- `clocks.max.gr` - Maximum graphics clock
- `clocks.max.mem` - Maximum memory clock
- `encoder.stats.sessionCount` - Active encoding sessions
- `clocks.throttle_reasons.*` - Throttling information

## Prompt Generation

### Template Customization

```python
def generate_prompt(target_tokens: int) -> str:
    base_text = "Your base instruction here. "
    repeat_text = (
        "Your repeating content pattern. "
        "This will be repeated to reach target token count. "
    )
```

**Domain-Specific Templates:**

**Code Generation:**
```python
base_text = "Implement the following function with comprehensive documentation. "
repeat_text = (
    "The function should handle edge cases including null inputs, empty arrays, "
    "type mismatches, and out-of-bounds conditions. Include unit tests, performance "
    "considerations, and integration patterns. "
)
```

**Medical Analysis:**
```python
base_text = "Analyze the following clinical case study in detail. "
repeat_text = (
    "Patient presents with symptoms requiring differential diagnosis. Consider "
    "laboratory values, imaging results, patient history, medication interactions, "
    "and evidence-based treatment protocols. "
)
```

**Legal Document:**
```python
base_text = "Review the following legal document for compliance. "
repeat_text = (
    "Examine contractual obligations, regulatory requirements, jurisdictional "
    "considerations, precedent cases, liability clauses, and risk mitigation strategies. "
)
```

## Visualization Configuration

### Chart Dimensions

```python
# In visualize_results()
if has_gpu_stats:
    fig = plt.figure(figsize=(26, 24))
else:
    fig = plt.figure(figsize=(24, 18))
```

**Adjust for Display:**
- **Presentation (16:9)**: `figsize=(24, 13.5)`
- **Print (A4)**: `figsize=(11.7, 8.3)`
- **High-res poster**: `figsize=(40, 30)`

### DPI Settings

```python
plt.savefig(filename, dpi=300, bbox_inches='tight')
```

**Quality Levels:**
- **Screen viewing**: `dpi=100`
- **Standard print**: `dpi=300`
- **High-quality print**: `dpi=600`

### Color Schemes

```python
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
```

**Alternative Palettes:**

**Colorblind-friendly:**
```python
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494']
```

**Grayscale:**
```python
colors = ['#000000', '#404040', '#808080', '#B0B0B0', '#D0D0D0', '#E8E8E8']
```

## Performance Tuning

### Thread Pool Size

Concurrent requests use Python threading. Adjust if needed:

```python
# Current implementation
for i in range(num_concurrent_users):
    t = threading.Thread(target=make_request, args=(...))
```

**For very high concurrency (50+ users):**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
    futures = [executor.submit(make_request, ...) for _ in range(num_concurrent_users)]
```

### Memory Management

For large benchmark runs, consider:

```python
import gc

# After each test
result = run_benchmark(...)
all_results.append(result)
gc.collect()  # Force garbage collection
```

## Advanced Configuration

### Custom Metrics

Add custom metric collection:

```python
class GPUMonitor:
    def get_gpu_stats(self) -> Optional[Dict]:
        # ... existing code ...
        
        # Add custom metric
        custom_metric = your_custom_function()
        
        return {
            # ... existing metrics ...
            'custom_metric': custom_metric
        }
```

### Multi-GPU Support

For multi-GPU systems:

```python
def get_gpu_stats(self, gpu_id: int = 0) -> Optional[Dict]:
    result = subprocess.run([
        'nvidia-smi',
        f'--id={gpu_id}',  # Specify GPU
        '--query-gpu=...',
        '--format=csv,noheader,nounits'
    ], ...)
```

### Remote Testing

Test remote vLLM servers:

```python
API_BASE_URL = "http://remote-server:8000"

# Add authentication if needed
def make_request(...):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.post(API_ENDPOINT, json=data, headers=headers, ...)
```

## Environment Variables

Create a `.env` file for configuration:

```bash
# .env
VLLM_SERVER_URL=http://localhost:8000
REQUEST_TIMEOUT=900
GPU_POLL_INTERVAL=1
OUTPUT_TOKENS=500
```

Load in script:

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv('VLLM_SERVER_URL', 'http://localhost:8000')
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 900))
```

## Configuration Best Practices

1. **Start Small**: Begin with minimal config, expand as needed
2. **Monitor Resources**: Watch GPU memory during initial runs
3. **Document Changes**: Keep notes on configuration modifications
4. **Version Control**: Track configuration changes in git
5. **Validate Settings**: Test configuration changes with short runs first

## Configuration Examples

### Development Testing
```python
context_lengths = [1000, 10000]
concurrent_users = [1]
output_tokens = 100
TEST_PAUSE_DURATION = 2
```

### Production Validation
```python
context_lengths = [32000, 64000, 128000]
concurrent_users = [5, 10, 20]
output_tokens = 500
TEST_PAUSE_DURATION = 5
REQUEST_TIMEOUT = 1200
```

### Research Benchmarking
```python
context_lengths = list(range(1000, 260000, 10000))  # Every 10K
concurrent_users = [1, 2, 4, 8, 16]
output_tokens = 1000
GPU_POLL_INTERVAL = 0.5
TEST_PAUSE_DURATION = 10
```

For questions about configuration, please open an issue on GitHub.
