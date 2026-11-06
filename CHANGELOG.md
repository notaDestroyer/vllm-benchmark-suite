# Changelog

## Unreleased

### Added
- **Generated GPU spec table, Ampere to Blackwell** — `scripts/gen_gpu_specs.py`
  derives `data/gpu_specs.json` from per-SKU SM count, boost clock, bus width
  and memory data rate with the same formulas as the live NVML path; 49 cards:
  A100 (SXM/PCIe/40GB), A30, A40, A10, A10G, RTX A6000/A5000/A4000, RTX 3060 to
  3090 Ti, L40S, L40, L4, RTX 6000/5000/4000 Ada, RTX 4060 to 4090, H100
  SXM/PCIe/NVL, H200, H20, B200, RTX PRO 6000/5000/4500 Blackwell, RTX 5060 to
  5090, T4. Entries record their inputs, their source (`derived` or
  `datasheet`) and a real nvidia-smi name; tests assert the committed JSON
  matches the generator and every name resolves to its own key.

### Fixed
- `gpu_specs.json`: RTX 6000 Ada was listed at the GeForce-halved rate
  (182 TFLOPS); the professional part runs full rate (364). L40 is now a
  separate entry at its real half rate instead of resolving to the L40S.
- GPU name matching: a `Ti` / `SUPER` suffix in the detected name now requires
  the same suffix in the table key, so a 3080 Ti no longer resolves to the 3080
  entry.

## v4.2.0 - Know Your GPU (2026-11-06)

The suite now learns the GPU from the GPU: roofline peaks are derived from the
device via NVML instead of trusted from a table, and a new Capacity & Ceilings
analysis answers "does it fit, how far can I push it, how close to the limit
did I get, what would a different quant buy, and is the card power-limited"
with numbers whose formulas and references are recorded in the output.

### Added
- **Capacity & Ceilings analysis** (`advisory.capacity`, terminal panel, HTML
  section, share lines, `capacity_*.png`, analyst-bundle facts): weights and KV
  footprint for the served quant; KV capacity in tokens read from the server
  (vLLM `cache_config_info`, SGLang `max_total_num_tokens`) or estimated from
  GPU memory; max users per context and matrix cells over capacity, checked
  before the run starts; roofline decode ceiling per cell
  (`N × BW / (W + N × KV × L)` capped by compute) with attainment and binding
  limit; prefill ceiling / MFU per cell; context slope with KV share of bytes;
  quant what-ifs (BF16 / FP8 / INT4 / NVFP4 / FP8 KV) on this GPU; operating
  point (measured knee vs critical batch); power truth (sustained clock, power
  cap and thermal throttle fractions, enforced limit); replicas-vs-TP advice
  for multi-GPU hosts; a sized `vllm serve` / SGLang command. Formulas and
  references (Williams et al. 2009; Pope et al. 2022; Kaplan et al. 2020;
  Databricks 2023) are recorded in `method`.
- nvidia-smi poller now samples `clocks.max.sm`, `power.limit` and
  `clocks_throttle_reasons.active`; cells carry `sustained_clock_pct`,
  `power_limit_w`, `power_cap_frac`, `thermal_frac`.
- SGLang server info carries `kv_capacity_tokens`, `gpu_memory_utilization`
  (mem_fraction_static) and `kv_cache_dtype`.

### Fixed
- Chart generation crashed on matplotlib ≥ 3.9 (`boxplot(labels=)` renamed to
  `tick_labels=`), aborting the run after the JSON was written; the main chart is
  now guarded like the others and covered by a test.
- **Device-derived GPU specs** — on an NVIDIA host the roofline bandwidth and
  dense BF16/FP8/FP4 peaks are computed from the card's own CUDA core count,
  clocks, bus width and compute capability via NVML (`nvidia-ml-py`, now a
  dependency), using per-architecture constants from Ampere to Blackwell and the
  GeForce FP32-accumulate half rate. `--gpu-spec auto|table|nvml` selects the
  source; `auto` cross-checks the curated table and overrides it when they
  disagree by more than 15%, reporting the discrepancy. Device facts and the
  resolved spec are stored in `system_info.gpu_spec`.
- FP4 support in the roofline: NVFP4 / MXFP4 / FP4 quantization is detected,
  weighs 0.5 B/param, and prefill MFU uses the FP4 peak.

### Fixed
- `gpu_specs.json`: the RTX PRO 6000 Blackwell entry used GeForce-halved
  tensor rates (251 / 503 TFLOPS); the professional part runs full rate, so it is
  now 503.8 BF16 / 1007.6 FP8 / 2015.2 FP4. FP4 added for the RTX 5090 (838).
- Fitness verdict no longer names the same profile as best and worst.

## v4.1.0 - Correctness & Review Fixes (2026-11-06)

First release since v3.1.1 on PyPI: v4.0.0 was tagged in the changelog but never
tagged in git or published, so this release carries the whole v4.0.0 feature set
below plus the fixes in this section.

### Fixed
- **Ported from PR #4 (thanks @tattrongvu)** — `save_result` no longer crashes when
  the GPU or model is reported as `None` (CPU-only clients); the nvidia-smi
  "not found" warning is printed once per process instead of on every poll;
  new `--tokenizer ID_OR_PATH` flag (or `VLLM_BENCH_TOKENIZER`) overrides the
  tokenizer used for prompt generation and token counting, for gated or
  locally-served models.
- **Multi-GPU hosts** — `nvidia-smi` prints one line per GPU; `SystemInfo` now
  parses per line, sums `total_vram_gb`, collapses identical names to
  `"Nx <name>"` and records `gpu_count` (fixes #2).
- **`--prompts-file` was a no-op** — the flag was stored but never read. Custom
  prompts are now validated up front and cycled across the concurrent requests
  of every cell; cells report `prompt_type=custom`, `prompt_source`,
  `prompts_file` and `num_custom_prompts`.
- **`--vs` always reported ties** — head-to-head looked for `*_samples` lists the
  engine never produced, so every comparison ran on n=1 point estimates. Cells
  now carry `latency_samples`, `ttft_samples`, `prefill_tps_samples`,
  `decode_tps_samples` and `request_tps_samples`; plain `prefill_tps` /
  `decode_tps` aliases are emitted. The latency comparison, previously sampled
  from raw latencies but labelled `latency_p99`, is now `avg_latency`; verdicts
  carry `point_a`/`point_b` beside `mean_a`/`mean_b`.
- **`--iterations > 1` degraded the analysis** — aggregation kept only four
  metrics. Every numeric field is now the median across iterations, sample
  lists are pooled, non-numeric fields are carried, and `n_iterations` is set.
- **Unreachable bottleneck classes** — `kv_capacity` / `interconnect` (and the
  waiting-requests `queue` path) read server signals that nothing attached to
  cells. `MetricsMonitor` now parses vLLM *and* SGLang metric names, polls during
  the cell and attaches peak `gpu_cache_usage_perc`, `num_requests_waiting`,
  `num_requests_running` and the `preemptions` delta; `tensor_parallel` is
  stamped on cells and passed to `classify_run`.
- **KL quality never reached the analyst report** — the bundle read
  `kl_divergence` while the KL mode emits `mean_kl` / `top1_agreement`. All
  numeric quality facts (probe CI, perplexity token counts, KL agreement) and
  the full `ScoreBreakdown` (new `to_dict()`) are now carried.
- **Wrong `head_dim` in the model KB** — `hidden // heads` gave 224 for
  Gemma-2-9B (actual 256), 144 for Gemma-2-27B (128) and 64 for Qwen3-30B-A3B /
  235B-A22B (128), skewing KV bytes, MBU and bottleneck verdicts. Every KB entry
  carries an explicit `head_dim`; DeepSeek-V3 uses MLA latent KV math
  (`layers × latent × bytes`, no K/V doubling), also honoured for live HF
  configs with `kv_lora_rank`.
- **`weight_bytes` ignored quantization** — now `total_params × bytes_per_param`
  of the served format (FP8/INT8 → 1 B, AWQ/GPTQ/INT4 → 0.5 B).
- **`publish.py`** — read `tensor_parallel_size` (ServerInfo emits
  `tensor_parallel`), raised on cells without `avg_latency` or with
  `ttft_estimate=None`, and labelled local time as UTC.
- **HTML report** — model names, levers, verdicts, diagnostics and system fields
  were interpolated unescaped and the results JSON could break the inline
  `<script>`; everything is escaped now. The footer reported a hard-coded
  v3.0.0 and the docstring claimed no external dependencies (Plotly loads from
  its CDN).
- **GPU baseline matching** — `"NVIDIA A100-SXM4-80GB"` and friends matched no
  scoring baseline or price. One token-based matcher
  (`core.hardware.gpu_names.match_gpu_key`) now serves scoring, pricing and the
  roofline spec table; auto-detected cost is multiplied by the GPU count.
- `gpu_spec` is initialised before the analysis stage so a failed analysis cannot
  leave it undefined for charts.
- **Review follow-ups** — the GPU matcher requires every designator token, so an
  RTX 3090 / 4080 / A6000 no longer resolves to the RTX 4090 baseline, price or
  spec; `--prompts-file` is honoured by `--traffic` mode; head-to-head compares
  per-request throughput as its own `request_tps` metric, never as a stand-in for
  the aggregate, and withholds a winner (`tie_reason = "direction_conflict"`)
  when sample means and point estimates disagree; auto-detected cost is billed
  for the GPUs the server occupies (TP × PP) rather than the whole host; the
  Prometheus parser tolerates timestamp suffixes and braces inside label values;
  a stalled metrics poller can no longer leak peaks into a later window;
  `publish` rejects NaN latency/TTFT via the shared `is_finite_number`; custom
  prompt token counts are memoised per file and sustained runs count only the
  prompts actually sent; the prompts file is parsed once.

### Added
- `roofline_*.png` and `bottleneck_map_*.png` are written with every charted run.
- `--leaderboard [DIR]` standalone mode writes `LEADERBOARD.md` from
  `result_*.json` files; `--detailed-summary` prints the per-cell tables.
- README sections for the composite GitHub Action and the Grafana dashboard.

### Changed
- The governing-verdict selection, the `N/A` formatter and the CI-overlap helpers
  each have a single implementation (`bottleneck.top_verdict`,
  `reports.formatting.na`, `statistics.point_ci` / `cis_overlap`). The verdict
  tie-break now prefers the highest-concurrency cell among equal confidence.
- SciPy import fallbacks (manual Welch t-test, bootstrap/normal approximations)
  were unreachable since SciPy became a hard dependency and are removed.
- The CSV keeps scalar columns only; sample lists and nested stats live in the JSON.

## v4.0.0 - Benchmark & Compare (2026-06-07)

The v4 release turns the suite from a vLLM-only benchmarker into a **model-aware
benchmark-and-compare** tool that profiles *and* explains performance across vLLM
and SGLang, with roofline-based bottleneck analysis, a statistics overhaul, and an
optional AI analyst report. Results JSON now carries `schema_version = "4.0"`.

### Added
- **Multiple backends** — `--backend auto|vllm|sglang` with auto-detection; SGLang
  is a first-class target alongside vLLM.
- **Second console alias** — `llm-bench` is now an entry point alongside `vllm-bench`.
- **Workload coverage** — `--workload auto|generative|embeddings|structured`. Auto
  picks the workload from the server task/model; embeddings and structured/function-
  calling have dedicated runners and metrics.
- **Model Intelligence** — MoE-vs-dense detection, active/total parameter accounting,
  GQA/MQA attention grouping, and KV-cache / VRAM math, emitted as `model_profile`.
- **Roofline & bottleneck engine** — Model Bandwidth Utilization (MBU) and Model
  FLOPs Utilization (MFU), the critical batch size B*, and per-cell governing-
  bottleneck classification (`bottlenecks`), plus a config `advisory`. New
  `--bottleneck-sweep` runs empirical prefill/decode roofline probes to locate the
  critical batch.
- **Application fitness** — workload-aware fitness verdict surfaced in the advisory.
- **Quality measurement** — `--quality off|probe|perplexity|kl` with `--quality-ref`;
  reported in its own `quality` section and never folded into the performance score.
  Perplexity/KL modes are backend-gated (require logprob/token-level support).
- **Statistics overhaul** — BCa bootstrap confidence intervals, Welch's t-test and
  Mann-Whitney U, Cohen's d and Cliff's delta effect sizes, Holm-Bonferroni multiple-
  comparison correction, and Wilson intervals for proportions. A "real difference"
  must be both statistically significant *and* practically significant.
- **Sharing & comparison** — `--share` writes a copy-paste Reddit/Markdown summary
  (`share_*.md`); a result card PNG; `--compare-quants FILE...` for Holm-corrected
  quant/model comparisons; and `--vs URL2` for a vLLM-vs-SGLang head-to-head A/B.
- **AI analyst report** — `--ai-report` with `--report-provider local|openai|claude`,
  `--report-llm-url`, `--report-model`, `--report-max-tokens`. All numbers are
  computed by the tool from the run's facts; the LLM only writes prose; a numeric
  verifier redacts any figure not present in the data, and on failure the tool falls
  back to a deterministic report. Optional Claude provider via
  `pip install vllm-benchmark-suite[report-claude]`.

### Changed
- **Timing** — switched to `time.perf_counter` throughout and split throughput into
  measured **prefill** (`prefill_tps`) and **decode** (`decode_tps`) phases instead
  of a fixed-percentage estimate.
- **Hardware layer** — refactored NVIDIA hardware detection and added a GPU spec KB
  (peak FLOPs, HBM bandwidth) used by the roofline math.
- **Runtime dependencies** — promoted `psutil` from a best-effort import to a hard
  dependency and declared `huggingface_hub`, `scipy`, and `jsonschema` (all imported
  at runtime). `anthropic` remains optional under the `report-claude` extra.
- **Results schema** — metadata now includes `schema_version`, `model_profile`,
  `bottlenecks`, `advisory`, and `quality`.

### Fixed
- Statistics now handle empty/degenerate samples without emitting `nan` (CIs and
  effect sizes degrade gracefully instead of poisoning downstream summaries).

---

## v3.0.0 - Professional Package (2026-04-16)

### Breaking Changes
- Restructured as installable Python package: `pip install vllm-benchmark-suite`
- CLI entry point is now `vllm-bench` (replaces running scripts directly)
- Removed v1 and v2 standalone scripts

### New Features
- **vLLM Score**: Composite benchmark score (0-10,000) with grade (S/A/B/C/D/F) — like Geekbench for vLLM
- **Auto-Diagnostics**: Plain-English analysis with actionable recommendations after every run
- **Interactive HTML Reports**: Self-contained Plotly-based reports shareable without dependencies
- **True TTFT Measurement**: SSE streaming-based Time-to-First-Token (not estimates)
- **Regression Detection**: `--compare previous.json` to catch performance regressions
- **Benchmark Presets**: `--quick` (5min), `--standard` (30min), `--thorough` (2hr)
- **Custom Prompts**: `--prompts-file my_prompts.jsonl` for production-realistic testing
- **Full CLI**: Non-interactive mode (`-y`), all parameters configurable via flags

### Architecture
- Proper Python package with `src/` layout and `pyproject.toml`
- Modular structure: `core/`, `analysis/`, `reports/`
- 18 unit tests covering scoring, diagnostics, and prompt generation
- GitHub Actions CI for Python 3.10-3.12
- Issue templates for bug reports and feature requests

---

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
