# vLLM Benchmark Suite

> A rigorous benchmarking tool for vLLM inference servers. Async load generation, statistical confidence intervals, plain-English diagnostics, and shareable reports.

[![PyPI](https://img.shields.io/pypi/v/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![Python](https://img.shields.io/pypi/pyversions/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/notadestroyer/vllm-benchmark-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/notadestroyer/vllm-benchmark-suite/actions)

---

## Quick Start

```bash
pip install vllm-benchmark-suite
vllm-bench --quick
```

Point it at your vLLM server and get a full performance profile in ~5 minutes.

---

## What It Does

- **True async load** — requests run concurrently via `aiohttp`, not threads. No GIL bottleneck.
- **Two load modes** — burst (all N requests simultaneously) and sustained RPS (token-bucket rate limiter).
- **True TTFT** — actual Time-to-First-Token via SSE streaming, not an estimate.
- **Accurate token counts** — uses `AutoTokenizer` from `transformers`, not `len(text) // 4`.
- **Statistical rigor** — run multiple iterations to get 95% confidence intervals, outlier detection, and CV warnings.
- **Cost analysis** — tokens per dollar, cost per 1M tokens, auto-detected from GPU name.
- **Composite score** — weighted 0–10,000 score across throughput, latency, efficiency, energy, and consistency.
- **Auto-diagnostics** — 10+ rule-based checks with plain-English recommendations.
- **Regression detection** — compare two runs and flag statistically meaningful changes.
- **Shareable reports** — self-contained HTML with Plotly charts, plus PNG and JSON/CSV.

---

## Benchmark Presets

| Preset | Time | Context Lengths | Concurrency | Prompt Types |
|--------|------|-----------------|-------------|--------------|
| `--quick` | ~5 min | 32K | 1, 4 | classic |
| `--standard` | ~30 min | 32K, 64K, 128K | 1, 4, 8, 16 | classic, deterministic |
| `--thorough` | ~2 hours | 32K–512K | 1, 4, 8, 16, 32 | all 4 types |

Or configure everything manually:

```bash
vllm-bench --context-lengths 32k,64k,128k --concurrency 1,4,8,16 --output-tokens 500
```

---

## Load Modes

### Burst (default)
Fires all `N` concurrent requests simultaneously. Tests peak throughput and how well vLLM handles queue pressure.

```bash
vllm-bench --standard --concurrency 1,4,8,16
```

### Sustained RPS
Sends requests at a steady rate for a fixed duration. Tests real-world latency behaviour under continuous load.

```bash
vllm-bench --rps 10 --duration 120
```

Produces per-time-bucket latency tracking (avg and P99 in 10-second windows), actual vs target RPS, and steady-state detection.

---

## Statistical Rigor

Single-run results have no confidence bounds. Use `--iterations` to get them:

```bash
vllm-bench --standard --iterations 5 --seed 42
```

With multiple iterations the tool runs each `(context, concurrency, prompt_type)` combination `N` times, then aggregates:

- **95% confidence intervals** on throughput, latency, TTFT (t-distribution via SciPy, bootstrap fallback)
- **IQR outlier detection** with transparent fence reporting
- **Coefficient of variation** warnings when variance is too high to trust results
- **Welch's t-test + Cohen's d** for regression comparisons

Summary table shows CI bounds:

```
Peak Throughput    1,500.3 tok/s  [1,480.1 – 1,520.5]   16u @ 32K
```

`--seed` sets `random.seed` and `numpy.random.seed` for reproducible prompt generation. The full environment fingerprint (kernel, CPU governor, GPU clocks, driver, package versions) is captured as a SHA-256 hash and printed at the end of every run.

---

## Cost Analysis

```bash
vllm-bench --standard --cost 2.21       # explicit $/hr
vllm-bench --standard                   # auto-detected from GPU name
```

Known GPU hourly rates (cloud on-demand):

| GPU | $/hr |
|-----|------|
| H100 | $4.00 |
| A100 80GB | $2.21 |
| A100 40GB | $1.80 |
| L40S | $1.50 |
| RTX 4090 | $0.74 |
| T4 | $0.53 |

Reported per test: cost per 1M tokens, total cost for that configuration.

---

## Prompt Strategies

Four strategies let you control prefix cache behaviour:

| Type | Cache behaviour | Use case |
|------|----------------|----------|
| `classic` | High cache hits | Realistic long-context workload |
| `deterministic` | Near-perfect cache hits | Best-case cache performance |
| `madlib` | Moderate cache misses | Mixed workload |
| `random` | Minimal cache hits | Worst-case / stress test |

Or use your own:

```bash
vllm-bench --prompts-file production_prompts.jsonl
```

JSONL format: one JSON object per line with a `"prompt"` key.

---

## vLLM Score (0–10,000)

A single composite number for easy comparison across runs and deployments.

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Throughput | 30% | Peak tokens/sec vs GPU reference |
| Latency | 25% | Best average latency (lower = better) |
| Efficiency | 20% | Tokens/sec per concurrent user |
| Energy | 15% | Tokens per watt |
| Consistency | 10% | Latency coefficient of variation |

Grades: **S** (9000+) · **A** (7500–8999) · **B** (6000–7499) · **C** (4000–5999) · **D** (2000–3999) · **F** (<2000)

GPU-specific reference baselines are built in for H100, A100, L40S, RTX 4090, T4, and others.

---

## Diagnostics

After every run, 10+ automated checks produce plain-English findings:

```
OK   Peak throughput 1,360 tok/s at 32K, 16 users
WARN GPU temperature peaked at 82°C — thermal throttling risk
WARN p99 latency is 5× average at 128K — likely request queuing
     Consider reducing max_num_seqs or enabling prefix caching
```

Checks include: request failure rate, latency variance, GPU utilisation, TTFT, batch scaling efficiency, cache effectiveness, memory pressure, temperature, and energy efficiency. When vLLM server info is available, config recommendations are included (prefix caching, tensor parallelism, quantization, `max_num_seqs`).

---

## Regression Detection

```bash
vllm-bench --standard --compare baseline.json
```

Compares results matched by `(context_length, concurrency, prompt_type)`. Flags changes against configurable thresholds (major: >15%, minor: 5–15%). With `--iterations`, uses Welch's t-test to distinguish real regressions from measurement noise.

---

## Output Files

Each run writes to `./outputs/` (override with `--output-dir`):

| File | Description |
|------|-------------|
| `benchmark_*.json` | All results + metadata, system info, environment fingerprint |
| `benchmark_*.csv` | Tabular results for spreadsheet analysis |
| `benchmark_*.png` | 5 publication-quality charts (300 DPI) |
| `benchmark_*.html` | Self-contained interactive report (Plotly, dark theme) |
| `result_*.json` | Standardised entry for community leaderboard (optional) |

### Charts

1. **Throughput vs Context Length** — line plot per concurrency level
2. **Latency Distribution** — box plot with P99 overlay
3. **TTFT Distribution** — with UX quality zones (green <200 ms, yellow <1 s, red >1 s)
4. **Throughput Heatmap** — context × concurrency grid
5. **GPU Utilization & Power** — dual-axis timeline

---

## Metrics Reference

**Throughput**: `tokens_per_second`, `requests_per_second`, `throughput_per_user`

**Latency**: `avg_latency`, `min_latency`, `max_latency`, `latency_p50/p90/p95/p99`

**TTFT**: `ttft_estimate`, `ttft_p50/p90/p95/p99`

**Inter-token latency**: `inter_token_latency`, `itl_p50/p90/p95/p99`

**GPU** (nvidia-smi): `avg_gpu_util`, `max_gpu_util`, `avg_mem_used`, `avg_temperature`, `avg_power`, `avg_gpu_clock`

**Energy**: `tokens_per_watt`, `watts_per_token`, `energy_joules`

**Cache** (vLLM `/metrics` endpoint): `cache_hit_rate`, `actual_prefill_time`, `actual_decode_time`

**Cost** (when available): `cost_per_hour`, `cost_per_1m_tokens`, `cost_total`

**Tokens**: `prompt_tokens`, `completion_tokens`, `total_tokens`

**Statistical** (with `--iterations > 1`): `*_ci_lower`, `*_ci_upper` for throughput, latency, and TTFT metrics

---

## CLI Reference

```
vllm-bench [OPTIONS]

Connection:
  --url URL              vLLM server URL (default: http://localhost:8000)
  --model NAME           Model name override (auto-detected)

Presets (mutually exclusive):
  --quick                ~5 min
  --standard             ~30 min
  --thorough             ~2 hours

Test Parameters:
  --context-lengths      Comma-separated, e.g. 32k,64k,128k or 1m
  --concurrency          Comma-separated, e.g. 1,4,8,16
  --output-tokens N      Max output tokens per request (default: 500)
  --prompt-type TYPE     classic|deterministic|madlib|random|all
  --prompts-file PATH    Custom prompts JSONL file

Load Mode:
  --rps FLOAT            Sustained requests-per-second mode
  --duration FLOAT       Duration for sustained RPS run (default: 120s)

Statistical Rigor:
  --iterations N         Iterations per config for confidence intervals (default: 1)
  --seed INT             Random seed for reproducibility

Cost:
  --cost FLOAT           GPU cost in USD/hr (auto-detected for known GPUs)

Behavior:
  -y, --non-interactive  Skip interactive prompts, use defaults
  --no-warmup            Skip model warmup
  --no-streaming         Disable streaming TTFT measurement

Output:
  --output-dir DIR       Output directory (default: ./outputs)
  --no-html              Skip HTML report
  --no-charts            Skip PNG charts

Traffic Simulation:
  --traffic TYPE         poisson|multiturn
  --target-rps FLOAT     Target RPS for traffic simulation (default: 2.0)
  --traffic-duration S   Duration in seconds (default: 60)
  --turns N              Turns per conversation for multiturn (default: 5)

Comparison:
  --compare FILE         Compare with previous results JSON
```

---

## Installation

```bash
# From PyPI
pip install vllm-benchmark-suite

# With uv
uv pip install vllm-benchmark-suite

# From source
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, a running vLLM server. NVIDIA GPU optional (required for GPU metrics).

---

## Contributing

```bash
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

Open an issue first to discuss significant changes.

---

## License

MIT
