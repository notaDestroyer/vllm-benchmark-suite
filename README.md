# vLLM Benchmark Suite

> Benchmark **and compare** vLLM and SGLang — model-aware, with roofline bottleneck analysis and an optional AI analyst report. Async load generation, statistical confidence intervals, plain-English diagnostics, and shareable reports.

[![PyPI](https://img.shields.io/pypi/v/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![Python](https://img.shields.io/pypi/pyversions/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/notadestroyer/vllm-benchmark-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/notadestroyer/vllm-benchmark-suite/actions)

---

## Quick Start

```bash
pip install vllm-benchmark-suite
vllm-bench --quick          # or: llm-bench --quick
```

Point it at your inference server and get a full performance profile in ~5 minutes. The CLI ships under two names — `vllm-bench` and `llm-bench` — which are interchangeable.

---

## What It Does

- **Benchmark and compare** — profile a server, then compare quantizations, models, or two endpoints head-to-head (vLLM vs SGLang).
- **Model-aware** — detects MoE vs dense, active/total parameters, and attention grouping, then does the KV-cache and VRAM math for you.
- **Roofline bottleneck analysis** — tells you whether each load point is memory-bandwidth, compute, or overhead bound, and what lever to pull.
- **True async load** — requests run concurrently via `aiohttp`, not threads. No GIL bottleneck.
- **True TTFT and split throughput** — actual Time-to-First-Token via SSE streaming, and measured prefill vs decode throughput (not a fixed-percentage estimate).
- **Accurate token counts** — uses `AutoTokenizer` from `transformers`, not `len(text) // 4`.
- **Statistical rigor** — BCa confidence intervals, hypothesis tests, effect sizes, and multiple-comparison correction.
- **Quality measurement** — probe / perplexity / KL-divergence modes, reported separately from performance.
- **Optional AI analyst report** — an LLM writes the narrative; every number is computed and verified by the tool.
- **Shareable output** — self-contained HTML, PNG charts, a copy-paste Markdown summary, and a result card image.

---

## Backends

```bash
vllm-bench --backend auto      # detect vLLM or SGLang from the server (default)
vllm-bench --backend vllm
vllm-bench --backend sglang
```

`auto` inspects the target server and selects the right client and capability probes. SGLang is a first-class target alongside vLLM, so the same matrix can be run against either engine.

---

## Workloads

```bash
vllm-bench --workload auto          # pick from the server task / model (default)
vllm-bench --workload generative    # text generation (TTFT, decode, throughput)
vllm-bench --workload embeddings    # single-forward-pass embedding throughput
vllm-bench --workload structured    # JSON / function-calling probes
```

`auto` chooses the workload based on the served model and task. Embeddings and structured/function-calling have dedicated runners and metrics; generation-only fields (e.g. TTFT, `decode_tps`) are omitted where they do not apply.

---

## Model Intelligence

From the server's model metadata the suite builds a **model profile** (`model_profile` in the JSON):

- **MoE vs dense** — and for mixture-of-experts, the **active** vs **total** parameter counts (only active params move per token, which is what bandwidth/FLOPs math should use).
- **Attention grouping** — GQA / MQA detection, which drives KV-cache size per token.
- **KV-cache & VRAM math** — bytes-per-token for the KV cache (scaled by KV heads, head dim, layers, and dtype) and an estimate of weight + KV memory, so you can reason about how much context fits.

These quantities feed directly into the roofline math below.

---

## Roofline & Bottleneck Analysis

The suite classifies every `(context, concurrency)` cell against a roofline model and reports the **governing bottleneck** (`bottlenecks` in the JSON) plus a config **advisory**.

- **MBU — Model Bandwidth Utilization.** During decode, generation is usually limited by how fast weights (and KV cache) can be streamed from HBM. MBU is the fraction of the GPU's peak HBM bandwidth you are actually using; low MBU at high batch suggests overhead or scheduling limits.
- **MFU — Model FLOPs Utilization.** During prefill, you are usually compute-bound. MFU is the fraction of the GPU's peak FLOPs you are achieving. **Caveat:** MFU uses the standard `2 · params` FLOPs-per-token approximation and is accurate to roughly **±15%** — treat it as a guide, not a billing-grade figure.
- **Critical batch size B\*.** The batch size where the workload transitions from memory-bandwidth bound (small batches) to compute bound (large batches), derived from the GPU's FLOPs-to-bandwidth ratio. Below B\* you are wasting compute; above it you are wasting bandwidth headroom.
- **Bottleneck classes.** Each cell is labelled memory-bandwidth bound, compute bound, or overhead/latency bound, with a confidence level and the recommended lever (e.g. raise batch, enable prefix caching, change parallelism).

Run empirical probes to find B\* on your actual hardware instead of relying on the analytic estimate:

```bash
vllm-bench --bottleneck-sweep
```

This sweeps prefill and decode probes across batch sizes and reports the empirical critical batch alongside the analytic one.

Two extra PNGs are written with every charted run: `roofline_*.png` (the hardware roofline with the ridge point and measured prefill/decode operating points) and `bottleneck_map_*.png` (context × concurrency, coloured by governing bottleneck, with B\* annotated).

The `kv_capacity`, `queue` and `interconnect` classes are driven by server-side signals that the suite samples from `/metrics` during each cell (vLLM and SGLang names are both understood): peak `gpu_cache_usage_perc`, peak `num_requests_waiting` / `num_requests_running` and the `preemptions` delta are attached to every result cell, and `tensor_parallel` is stamped from the server's reported parallelism.

---

## Quality Measurement

Quality is measured in its own section and **never** folded into the performance score.

```bash
vllm-bench --quality off            # default
vllm-bench --quality probe          # rubric-style correctness probes -> score /100
vllm-bench --quality perplexity     # perplexity over a built-in corpus
vllm-bench --quality kl --quality-ref http://reference:8000   # KL vs a reference
```

`perplexity` and `kl` are **backend-gated**: they require the server to expose the token-level / logprob information they depend on, and are skipped (with a reason) when it is unavailable. `--quality-ref` supplies the reference endpoint that `kl` compares against.

---

## Statistics

Single-run numbers have no error bars. Use `--iterations` to get them:

```bash
vllm-bench --standard --iterations 5 --seed 42
```

The statistics layer provides:

- **BCa bootstrap confidence intervals** (bias-corrected and accelerated) on throughput, latency, and TTFT.
- **Hypothesis tests** — Welch's t-test (unequal variances) and the Mann-Whitney U test (rank-based, distribution-free).
- **Effect sizes** — Cohen's *d* and Cliff's delta, so you know not just *whether* something changed but *how much*.
- **Holm-Bonferroni correction** when many configurations are compared at once (e.g. `--compare-quants`).
- **Wilson intervals** for proportions such as success rate.

The governing rule: a **real difference is one that is both statistically significant *and* practically significant** (a non-trivial effect size). A tiny but "significant" delta from a huge sample is reported as noise, not a win. Degenerate/empty samples degrade gracefully instead of emitting `nan`.

With `--iterations N` every numeric metric in a cell (prefill/decode split, percentiles, GPU telemetry, cost, failure counts) is the median across iterations, so bottleneck, fitness and scoring see the same schema as a single run; the four headline metrics additionally carry `*_ci_lower`/`*_ci_upper`, a `*_stats` dict and a cross-iteration `*_samples` list, and `n_iterations` records the count.

`--seed` fixes `random.seed` and `numpy.random.seed`. A full environment fingerprint (kernel, CPU governor, GPU clocks, driver, package versions) is captured as a SHA-256 hash and printed at the end of every run.

---

## Sharing & Comparison

```bash
vllm-bench --standard --share                          # write a copy-paste Markdown summary
vllm-bench --compare-quants fp8.json awq.json gptq.json  # compare runs (Holm-corrected)
vllm-bench --standard --vs http://other-server:8000    # vLLM-vs-SGLang head-to-head A/B
```

- **`--share`** writes `share_*.md`, a Reddit/forum-ready Markdown summary you can paste directly.
- **Result card** — a shareable result-card PNG (`result_card.png`) is available via the `render_result_card` reporting API.
- **`--compare-quants FILE...`** compares multiple result JSONs across quantizations/models, with Holm-corrected significance and a `quant_compare_*.png` chart.
- **`--vs URL2`** re-runs the identical matrix against a second endpoint and reports a head-to-head A/B (`head_to_head_*.json`) — ideal for vLLM vs SGLang on the same model and hardware. The comparison runs on the per-request sample lists each cell carries (`latency_samples`, `ttft_samples`, `prefill_tps_samples`, `decode_tps_samples`, `request_tps_samples`), or on cross-iteration samples when `--iterations` is used, so Welch / Mann-Whitney tests have real spread. Metrics compared: `tokens_per_second`, `ttft_estimate`, `prefill_tps`, `decode_tps`, `avg_latency`. A winner is declared only when the difference is statistically and practically significant after Holm correction and the bootstrap CIs do not overlap; otherwise the cell is a tie.
- **`--leaderboard [DIR]`** builds `LEADERBOARD.md` from the `result_*.json` entries in a directory (default `./outputs`) without touching a server.

---

## AI Analyst Report

```bash
vllm-bench --standard --ai-report                              # local provider (the benchmarked server)
vllm-bench --standard --ai-report --report-provider openai --report-llm-url http://llm:8000
vllm-bench --standard --ai-report --report-provider claude --report-model claude-opus-4-8
```

The report turns the run's computed facts into a readable analyst writeup. Off by default; enable with `--ai-report`.

**Determinism boundary — read this.** Every number in the report is computed by the tool from the run's data. **The LLM only writes prose.** A numeric verifier scans the generated text and **redacts any figure that is not present in the underlying data**, so the model cannot invent or "round" a metric. If generation fails or the output cannot be verified, the tool **falls back to a deterministic, template-based report**. The narrative is advisory; the numbers are ground truth.

Flags:

| Flag | Meaning |
|------|---------|
| `--ai-report` | Enable the analyst report (off by default) |
| `--report-provider local\|openai\|claude` | `local` = the benchmarked server, `openai` = any OpenAI-compatible URL, `claude` = Anthropic |
| `--report-llm-url URL` | Endpoint for the `local` / `openai` providers |
| `--report-model NAME` | Model name (Claude defaults to `claude-opus-4-8`) |
| `--report-max-tokens N` | Generation budget (default: 8000) |

The Claude provider is optional. Install it with:

```bash
pip install vllm-benchmark-suite[report-claude]
```

The report is written to `report_*.md` and embedded into the HTML report.

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
Fires all `N` concurrent requests simultaneously. Tests peak throughput and how well the server handles queue pressure.

```bash
vllm-bench --standard --concurrency 1,4,8,16
```

### Sustained RPS
Sends requests at a steady rate for a fixed duration. Tests real-world latency behaviour under continuous load.

```bash
vllm-bench --rps 10 --duration 120
```

Produces per-time-bucket latency tracking (avg and P99 in windows), actual vs target RPS, and steady-state detection.

---

## Cost Analysis

```bash
vllm-bench --standard --cost 2.21       # explicit $/hr
vllm-bench --standard                   # auto-detected from GPU name
```

Reported per test: cost per 1M tokens and total cost for that configuration. Known cloud on-demand rates are built in for H100, A100 (80/40GB), L40S, RTX 4090, T4, and others; the auto-detected rate is per GPU and is multiplied by the detected GPU count. GPU names are matched by token (so `NVIDIA A100-SXM4-80GB`, `NVIDIA H100 80GB HBM3` and `4x NVIDIA H100 80GB HBM3` all resolve), with the same matcher used for the score baselines and the roofline spec table.

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
vllm-bench --prompts-file production_prompts.jsonl --concurrency 1,4,8
```

JSONL format: one JSON object per line with a `"prompt"` key. The file is validated before the server is touched. Prompts are cycled across the concurrent requests of every cell (request *i* gets prompt `i mod N`), the cell's `prompt_type` is reported as `custom`, and `prompt_source`, `prompts_file` and `num_custom_prompts` are recorded in each result. The `--context-lengths` axis only labels the matrix in this mode: the real prompt length is whatever your file contains and is reported as `avg_prompt_tokens`.

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

After every run, automated checks produce plain-English findings:

```
OK   Peak throughput 1,360 tok/s at 32K, 16 users
WARN GPU temperature peaked at 82°C — thermal throttling risk
WARN p99 latency is 5× average at 128K — likely request queuing
     Consider reducing max_num_seqs or enabling prefix caching
```

Checks include request failure rate, latency variance, GPU utilisation, TTFT, batch scaling efficiency, cache effectiveness, memory pressure, temperature, and energy efficiency. When server info is available, config recommendations (prefix caching, parallelism, quantization, `max_num_seqs`) are included.

---

## Regression Detection

```bash
vllm-bench --standard --compare baseline.json
```

Compares results matched by `(context_length, concurrency, prompt_type)` and flags changes against thresholds. With `--iterations`, uses the statistics layer to distinguish real regressions from measurement noise.

---

## Output Files

Each run writes to `./outputs/` (override with `--output-dir`):

| File | Description |
|------|-------------|
| `benchmark_*.json` | All results + metadata (`schema_version`, system/server info, environment fingerprint, `model_profile`, `bottlenecks`, `advisory`, `quality`) |
| `benchmark_*.csv` | Tabular results for spreadsheet analysis (scalar columns only; sample lists live in the JSON) |
| `benchmark_*.png` | Publication-quality charts (300 DPI) |
| `roofline_*.png` | Roofline plot with ridge point and measured operating points |
| `bottleneck_map_*.png` | Context × concurrency governing-bottleneck map |
| `benchmark_*.html` | Self-contained interactive report (Plotly), embeds the analyst report when generated |
| `share_*.md` | Copy-paste Markdown summary (`--share`) |
| `report_*.md` | AI analyst report (`--ai-report`) |
| `result_card.png` | Shareable result card image (via `render_result_card`) |
| `quant_compare_*.png` | Quantization/model comparison chart (`--compare-quants`) |
| `head_to_head_*.json` | Head-to-head A/B results (`--vs`) |
| `result_*.json` | Standardised entry for community leaderboard (optional) |
| `LEADERBOARD.md` | Markdown leaderboard built from `result_*.json` files (`--leaderboard`) |

---

## Metrics Reference

**Throughput**: `tokens_per_second`, `requests_per_second`, `throughput_per_user`

**Split throughput**: `prefill_tps` / `decode_tps` per request, aggregated as `prefill_tps_mean`, `prefill_tps_p50/p90/p99`, `decode_tps_mean`, `decode_tps_p50/p90/p99`

**Latency**: `avg_latency`, `min_latency`, `max_latency`, `latency_p50/p90/p95/p99`

**TTFT**: `ttft_estimate`, `ttft_p50/p90/p95/p99`

**Inter-token latency**: `inter_token_latency`, `itl_p50/p90/p95/p99`

**Roofline** (`bottlenecks`, `advisory`): `mbu`, `mfu`, critical batch B\*, governing-bottleneck class + confidence + lever

**Model profile** (`model_profile`): `is_moe`, active/total params, attention grouping (GQA/MQA), KV bytes per token, estimated VRAM

**Quality** (`quality` section): mode, status, and `score` (probe) / perplexity / KL as applicable

**GPU** (nvidia-smi): `avg_gpu_util`, `max_gpu_util`, `avg_mem_used`, `avg_temperature`, `avg_power`, `avg_gpu_clock`. On multi-GPU hosts `system_info.gpu_name` is collapsed to `"Nx <name>"`, `gpu_count` is recorded and `total_vram_gb` is summed across GPUs.

**Energy**: `tokens_per_watt`, `watts_per_token`, `energy_joules`

**Cache & scheduler** (server `/metrics`, sampled during the cell): `cache_hit_rate`, `actual_prefill_time`, `actual_decode_time`, peak `gpu_cache_usage_perc`, peak `num_requests_waiting`, peak `num_requests_running`, `preemptions` (delta), `tensor_parallel`

**Samples** (per request, for A/B statistics): `latency_samples`, `ttft_samples`, `prefill_tps_samples`, `decode_tps_samples`, `request_tps_samples`

**Prompt provenance**: `prompt_source` (`generated`/`file`), `prompts_file`, `num_custom_prompts`

**Cost** (when available): `cost_per_hour`, `cost_per_1m_tokens`, `cost_total`

**Tokens**: `prompt_tokens`, `completion_tokens`, `total_tokens`

**Statistical** (with `--iterations > 1`): `*_ci_lower`, `*_ci_upper` (BCa) for throughput, latency, and TTFT

---

## CLI Reference

```
vllm-bench [OPTIONS]          (also available as: llm-bench)

Connection:
  --url URL              Server URL (default: http://localhost:8000)
  --model NAME           Model name override (auto-detected)
  --backend TYPE         auto|vllm|sglang (default: auto)

Presets (mutually exclusive):
  --quick                ~5 min
  --standard             ~30 min
  --thorough             ~2 hours

Test Parameters:
  --context-lengths      Comma-separated, e.g. 32k,64k,128k or 1m
  --concurrency          Comma-separated, e.g. 1,4,8,16
  --output-tokens N      Max output tokens per request (default: 500)
  --prompt-type TYPE     classic|deterministic|madlib|random|all
  --prompts-file PATH    Custom prompts JSONL file (cycled across requests; prompt_type=custom)
  --rps FLOAT            Sustained requests-per-second mode
  --duration FLOAT       Duration for sustained RPS run (default: 120s)
  --iterations N         Iterations per config for confidence intervals (default: 1)
  --seed INT             Random seed for reproducibility

Workload:
  --workload TYPE        auto|generative|embeddings|structured (default: auto)

Quality:
  --quality MODE         off|probe|perplexity|kl (default: off)
  --quality-ref URL      Reference endpoint for --quality kl

Analysis:
  --bottleneck-sweep     Run prefill/decode roofline probes for the critical batch
  --detailed-summary     Print the full per-cell summary tables after the run
  --leaderboard [DIR]    Standalone: write LEADERBOARD.md from result_*.json in DIR and exit

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

Sharing & Comparison:
  --share                Write a copy-paste Markdown summary (share_*.md)
  --compare FILE         Compare with a previous results JSON (regression detection)
  --compare-quants FILE... Compare multiple result JSONs (Holm-corrected)
  --vs URL2              Head-to-head A/B against a second endpoint

AI Analyst Report:
  --ai-report            Generate an LLM analyst report from computed facts
  --report-provider P    local|openai|claude (default: local)
  --report-llm-url URL   LLM endpoint for local/openai providers
  --report-model NAME    Report model (Claude defaults to claude-opus-4-8)
  --report-max-tokens N  Generation budget (default: 8000)
```

---

## GitHub Action

The repository ships a composite action (`action.yml`) that installs the suite, runs a preset against a server reachable from the runner, uploads the `benchmark-results/` directory as an artifact and exposes the score, grade, peak throughput and results path as outputs. It can also fail the job on major regressions against a stored baseline.

```yaml
- uses: notadestroyer/vllm-benchmark-suite@main
  id: bench
  with:
    vllm-url: http://localhost:8000
    preset: quick                       # quick | standard | thorough
    compare-with: baseline/benchmark.json   # optional regression baseline
    fail-on-regression: "true"
- run: echo "Score ${{ steps.bench.outputs.score }} (${{ steps.bench.outputs.grade }})"
```

Inputs: `vllm-url`, `preset`, `context-lengths`, `concurrency`, `output-tokens`, `prompt-type`, `compare-with`, `fail-on-regression`, `python-version`. Outputs: `score`, `grade`, `peak-throughput`, `results-json`.

---

## Grafana Dashboard

`grafana/dashboard.json` is an importable Grafana dashboard for a Prometheus data source scraping vLLM's `/metrics`. It covers throughput, request latency, TTFT, inter-token latency, concurrent requests, KV-cache usage, prefix-cache hit rate, GPU utilisation / memory / temperature / power, request rate, active LoRA adapters and average prefill / decode time. Import it via *Dashboards → New → Import* and select your Prometheus data source.

---

## Installation

```bash
# From PyPI
pip install vllm-benchmark-suite

# With the optional Claude analyst provider
pip install vllm-benchmark-suite[report-claude]

# With uv
uv pip install vllm-benchmark-suite

# From source
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, a running vLLM or SGLang server. NVIDIA GPU optional (required for GPU metrics).

---

## Contributing

```bash
git clone https://github.com/notadestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

Open an issue first to discuss significant changes.

---

## License

MIT
