# vLLM Benchmark Suite

> The most comprehensive benchmarking tool for vLLM inference servers. Built by a vLLM user, for vLLM users.

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

That's it. Point it at your vLLM server and get a complete performance profile in 5 minutes.

---

## Why This Tool?

Every vLLM operator asks the same questions: *Is my setup fast enough? Where's the bottleneck? Did that config change make things better or worse?*

I built this because I was tired of writing one-off scripts to answer those questions. This tool gives you a **single benchmark score**, **plain-English diagnostics**, and **shareable HTML reports** — everything you need to understand, optimize, and communicate your vLLM deployment's performance.

---

## Features

### vLLM Score (0-10,000)
A single composite number — like Geekbench for vLLM. Compare deployments, track improvements, share results. Weighted across throughput, latency, efficiency, energy, and consistency.

### Auto-Diagnostics
After every run, get plain-English analysis: *"Your p99 latency is 5x your average at 128K context — this usually means request queuing. Try reducing max_num_seqs."* Not just numbers — actionable recommendations.

### True TTFT Measurement
Uses SSE streaming to measure actual Time-to-First-Token, not estimates. Know exactly how long users wait before they see output.

### Interactive HTML Reports
Self-contained HTML files with interactive Plotly charts. Share with your team, attach to tickets, email to your manager. Dark-themed, professional, no dependencies to view.

### Regression Detection
```bash
vllm-bench --standard --compare last_week.json
```
Automatically flags throughput drops and latency increases after vLLM upgrades, config changes, or infrastructure moves.

### 25+ Performance Metrics
- **Throughput**: tokens/sec, requests/sec, tokens/sec/user, batch scaling efficiency
- **Latency**: avg, min, max, P50/P90/P95/P99, TTFT, inter-token latency
- **GPU**: utilization, memory, temperature, power draw, clock frequencies
- **Energy**: tokens/watt, watts/token/user, total watt-hours consumed
- **Cache**: prefix cache hit rate, actual prefill/decode time separation

### 13+ Publication-Quality Charts
Throughput landscapes, latency/throughput heatmaps, TTFT with UX quality zones, inter-token latency, batch scaling efficiency, decode speed, power draw, prompt type comparisons, cache hit rate heatmaps.

### 4 Prompt Strategies
Test cache behavior with different prompt types:
- **Classic**: Deterministic cybersecurity text (high cache hits)
- **Deterministic**: Tokenizer-aware repetitive story (perfect cache hits)
- **Madlib**: Random word injection (moderate cache misses)
- **Random**: Fully random text (minimal cache hits)

### Custom Prompts
```bash
vllm-bench --prompts-file my_production_prompts.jsonl
```
Test with your actual production prompts for realistic performance numbers.

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

## CLI Reference

```
vllm-bench [OPTIONS]

Connection:
  --url URL              vLLM server URL (default: http://localhost:8000)
  --model NAME           Model name override (auto-detected)

Presets:
  --quick                Quick benchmark (~5 min)
  --standard             Standard benchmark (~30 min)
  --thorough             Thorough benchmark (~2 hours)

Test Parameters:
  --context-lengths      Comma-separated (e.g. 32k,64k,128k)
  --concurrency          Comma-separated (e.g. 1,4,8,16)
  --output-tokens N      Max output tokens (default: 500)
  --prompt-type TYPE     classic|deterministic|madlib|random|all
  --prompts-file PATH    Custom prompts JSONL file

Behavior:
  -y, --non-interactive  Skip interactive prompts
  --no-warmup            Skip model warmup
  --no-streaming         Disable streaming TTFT measurement

Output:
  --output-dir DIR       Output directory (default: ./outputs)
  --no-html              Skip HTML report
  --no-charts            Skip PNG charts

Comparison:
  --compare FILE         Compare with previous results JSON
```

---

## Sample Output

```
┌──────────────────────────────────────┐
│ vLLM Benchmark Suite                 │
│ v3.0.0                               │
└──────────────────────────────────────┘

  vLLM Benchmark Score: 7,234 / 10,000  (Grade: A)

  Throughput  (30%) ████████████████████░░░░░░░░░░ 6,800
  Latency     (25%) ██████████████████████████░░░░ 8,500
  Efficiency  (20%) ███████████████████████░░░░░░░ 7,650
  Energy      (15%) █████████████████░░░░░░░░░░░░░ 5,800
  Consistency (10%) █████████████████████████████░ 9,200

Diagnostics:
  OK Excellent throughput: 1,360 tokens/sec at 32K with 16 users
  OK All metrics look healthy
  WARNING GPU temperature peaked at 82°C

Performance Highlights:
  Peak Throughput    1,360.4 tok/s    16u @ 32K
  Best Efficiency      340.1 tok/s/user  1u @ 32K
  Lowest Latency        0.42s          1u @ 32K

Outputs:
  JSON: ./outputs/benchmark_Llama-3-70B_20240115_143022.json
  CSV:  ./outputs/benchmark_Llama-3-70B_20240115_143022.csv
  Charts: ./outputs/benchmark_Llama-3-70B_20240115_143022.png
  HTML: ./outputs/benchmark_Llama-3-70B_20240115_143022.html
```

---

## Installation

```bash
# From PyPI
pip install vllm-benchmark-suite

# Or with uv (faster)
uv pip install vllm-benchmark-suite

# From source
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
```

## Requirements

- **Python 3.10+**
- **A running vLLM server** (local or remote)
- **NVIDIA GPU** (for GPU metrics — benchmarking works without one, but you won't get GPU telemetry)

## Output Files

Each benchmark run generates:

| File | Description |
|------|-------------|
| `benchmark_*.json` | Complete results with metadata, system info, and all metrics |
| `benchmark_*.csv` | Tabular results for spreadsheet analysis |
| `benchmark_*.png` | 13+ publication-quality matplotlib charts (300 DPI) |
| `benchmark_*.html` | Interactive HTML report with Plotly charts |

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

## License

MIT
