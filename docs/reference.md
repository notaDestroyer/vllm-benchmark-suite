# Reference

Everything the README summarises, in full. For the tour, see the [README](../README.md).

- [CLI flags](#cli-flags)
- [Output files](#output-files)
- [Result JSON layout](#result-json-layout)
- [Metrics in each cell](#metrics-in-each-cell)
- [Bottleneck classes and levers](#bottleneck-classes-and-levers)
- [Score](#score)
- [Statistics](#statistics)
- [Head-to-head](#head-to-head)
- [Quant / model comparison](#quant--model-comparison)
- [Custom prompts](#custom-prompts)
- [Quality modes](#quality-modes)
- [Regression detection](#regression-detection)
- [AI analyst report](#ai-analyst-report)
- [GitHub Action](#github-action)

## CLI flags

```
vllm-bench [OPTIONS]          (also installed as: llm-bench)

Connection:
  --url URL              Server URL (default: http://localhost:8000)
  --model NAME           Model name override (auto-detected from /v1/models)
  --tokenizer ID|PATH    Tokenizer for prompt generation and token counting
                         (default: the served model name; env VLLM_BENCH_TOKENIZER)
  --backend TYPE         auto|vllm|sglang (default: auto)

Presets (mutually exclusive):
  --quick                32K; 1,4 users; classic; 150 output tokens (~5 min)
  --standard             32K/64K/128K; 1,4,8,16 users; classic+deterministic; 500 tokens (~30 min)
  --thorough             32K-512K; 1-32 users; all four prompt types; 500 tokens (~2 h)

Test parameters:
  --context-lengths      Comma-separated, e.g. 32k,64k,128k or 1m
  --concurrency          Comma-separated, e.g. 1,4,8,16
  --output-tokens N      Max output tokens per request (default: 500; --quick uses 150)
  --prompt-type TYPE     classic|deterministic|madlib|random|all (comma-separated allowed)
  --prompts-file PATH    Custom prompts JSONL (cycled across requests; prompt_type=custom)
  --rps FLOAT            Sustained requests-per-second mode
  --duration FLOAT       Duration in seconds for --rps (default: 120)
  --iterations N         Repeat every cell N times for confidence intervals (default: 1)
  --seed INT             Seeds random and numpy.random

Workload:
  --workload TYPE        auto|generative|embeddings|structured (default: auto)

Quality:
  --quality MODE         off|probe|perplexity|kl (default: off)
  --quality-ref URL      Reference endpoint for --quality kl

Analysis:
  --gpu-spec MODE        auto|table|nvml: where roofline peaks come from (default auto:
                         derive from the local GPU via NVML, cross-check the built-in table)
  --bottleneck-sweep     Run prefill/decode roofline probes to measure the critical batch
  --detailed-summary     Print the full per-cell summary tables after the run
  --leaderboard [DIR]    Standalone: write LEADERBOARD.md from result_*.json in DIR (default ./outputs) and exit

Cost:
  --cost FLOAT           GPU cost in USD/hr (auto-detected for known GPUs)

Behaviour:
  -y, --non-interactive  Skip interactive prompts, use defaults
  --no-warmup            Skip the warmup request
  --no-streaming         Non-streaming requests (no TTFT, no prefill/decode split)

Output:
  --output-dir DIR       Output directory (default: ./outputs)
  --no-html              Skip the HTML report
  --no-charts            Skip PNG charts

Traffic simulation:
  --traffic TYPE         poisson|multiturn
  --target-rps FLOAT     Target requests per second (default: 2.0)
  --traffic-duration S   Duration in seconds (default: 60)
  --turns N              Turns per conversation for multiturn (default: 5)

Sharing and comparison:
  --share                Write share_*.md and result_card.png
  --compare FILE         Regression check against a previous benchmark JSON
  --compare-quants F...  Standalone: compare two or more benchmark JSONs and exit
  --vs URL2              Replay the matrix against a second endpoint (head-to-head)

AI analyst report:
  --ai-report            Generate the LLM analyst report
  --report-provider P    local|openai|claude (default: local)
  --report-llm-url URL   Endpoint for local/openai providers
  --report-model NAME    Model for the provider (Claude defaults to claude-opus-4-8)
  --report-max-tokens N  Generation budget (default: 8000)
```

With no preset and no explicit matrix flags, the CLI asks interactively unless `-y` is given.

## Output files

All in `--output-dir` (default `./outputs/`):

| File | Written when | Contents |
|------|--------------|----------|
| `benchmark_<model>_<ts>.json` | always | Everything: metadata, every cell, model profile, bottlenecks, advisory, quality |
| `benchmark_<model>_<ts>.csv` | always | One row per cell, scalar columns only |
| `benchmark_<model>_<ts>.png` | unless `--no-charts` | Five-panel chart: throughput, latency, TTFT, heatmap, scaling |
| `roofline_<ts>.png` | unless `--no-charts` | Hardware roofline with ridge point and measured operating points |
| `bottleneck_map_<ts>.png` | unless `--no-charts` | Context × concurrency map coloured by governing bottleneck |
| `benchmark_<model>_<ts>.html` | unless `--no-html` | Interactive report. Data is embedded; the charts load Plotly from its CDN |
| `result_<gpu>_<model>_<ts>.json` | always | Standardised leaderboard entry |
| `share_<ts>.md` | `--share` | Reddit/forum Markdown summary |
| `result_card.png` | `--share` | 1200×630 result card |
| `report_<model>_<ts>.md` | `--ai-report` | Analyst report |
| `head_to_head_<ts>.json` | `--vs` | Per-cell, per-metric verdicts |
| `quant_compare_<ts>.json/.png` | `--compare-quants` | Comparison verdicts and chart |
| `LEADERBOARD.md` | `--leaderboard` | Table built from `result_*.json` |

## Result JSON layout

```
{
  "metadata": {
    "schema_version": "4.0",
    "timestamp", "benchmark_duration",
    "system_info":  { gpu_name, gpu_count, total_vram_gb, cuda_version, driver_version, ... },
    "server_info":  { backend, backend_version, model_name, max_model_len, quantization,
                      tensor_parallel, pipeline_parallel, max_num_seqs, prefix_caching, task, ... },
    "environment":  { fingerprint, kernel, cpu governor, clocks, package versions, ... },
    "configuration": { context_lengths, concurrent_users, output_tokens, prompt_types, streaming, iterations, seed },
    "model_profile": { name, family, is_moe, num_experts, experts_per_tok, total_params, active_params,
                       num_layers, hidden_size, num_attention_heads, num_kv_heads, head_dim, attention_type,
                       kv_bytes_per_token, weight_bytes, confidence, source, notes },
    "bottlenecks":  [ { cell, primary, contributions, mbu, mfu, critical_batch, headroom_to_ridge,
                        prefill_time_frac, lever, confidence, notes }, ... ],
    "advisory":     { latency_optimal, throughput_optimal, explanation, tips, confidence,
                      fitness: { verdict, profiles: { <profile>: { grade, limiting_factor, detail } } },
                      bottleneck_sweep (with --bottleneck-sweep) },
    "quality":      { mode, status, ... } | null
  },
  "results": [ <cell>, ... ]     // raw cells; with --iterations > 1 one per iteration, each tagged "iteration"
}
```

`model_profile.source` is `hf_live` (config fetched from the Hub), `kb` (built-in table), `server` or `name` (heuristics), and `confidence` is `confirmed`, `inferred` or `heuristic`.

## Metrics in each cell

Keys are present only when measurable. Non-streaming runs have no TTFT or prefill/decode split; embeddings cells have no generation metrics.

**Identity**: `context_length`, `concurrent_users`, `prompt_type`, `iteration`, `prompt_source` (`generated`/`file`), `prompts_file`, `num_custom_prompts`, `tensor_parallel`

**Throughput**: `tokens_per_second` (aggregate: completion tokens / wall time), `requests_per_second`, `throughput_per_user`

**Prefill / decode split** (streaming only): `prefill_tps`, `prefill_tps_mean`, `prefill_tps_p50/p90/p99`, `decode_tps`, `decode_tps_mean`, `decode_tps_p50/p90/p99`, `pp_tg_source`

**Latency**: `avg_latency`, `std_latency`, `min_latency`, `max_latency`, `latency_p50/p90/p95/p99`

**TTFT** (streaming): `ttft_estimate` (mean), `ttft_p50/p90/p95/p99`. In non-streaming mode `ttft_estimate` is 10 percent of mean latency and is labelled as an estimate.

**Inter-token latency**: `inter_token_latency`, `itl_p50/p90/p95/p99`

**Per-request samples** (lists, for the comparison statistics): `latency_samples`, `ttft_samples`, `prefill_tps_samples`, `decode_tps_samples`, `request_tps_samples` (completion tokens / duration per request)

**Counts**: `successful`, `failed`, `total_requests`, `avg_prompt_tokens`, `avg_completion_tokens`

**GPU** (nvidia-smi on the host running the benchmark): `avg_gpu_util`, `max_gpu_util`, `avg_mem_used`, `max_mem_used`, `avg_temperature`, `max_temperature`, `avg_power`, `max_power`, `avg_gpu_clock`, `max_gpu_clock`, `avg_mem_clock`; per-GPU breakdown under `per_gpu` on multi-GPU hosts

**Energy**: `tokens_per_watt`, `watts_per_token`, `energy_joules`, `energy_per_token`

**Server `/metrics`** (sampled during the cell): `cache_hit_rate`, `cache_queries_delta`, `cache_hits_delta`, `actual_prefill_time`, `actual_decode_time`, `prefill_speed`, peak `gpu_cache_usage_perc` (0 to 1), peak `num_requests_waiting`, peak `num_requests_running`, `preemptions` (delta)

**Cost** (with a known `$/hr`): `cost_per_hour`, `cost_per_1m_tokens`, `cost_total`

**Sustained mode only**: `mode`, `target_rps`, `actual_rps`, `duration_seconds`, `latency_over_time` (10-second buckets with avg, p99, count)

**With `--iterations > 1`** the cells handed to analysis and reports are aggregated: every numeric field is the median across iterations, sample lists are pooled, and `tokens_per_second`, `avg_latency`, `ttft_estimate`, `throughput_per_user` additionally carry `*_ci_lower`/`*_ci_upper` (95 percent Student-t interval), `*_stats` (mean, median, std, cv, percentiles, IQR, outliers) and `*_samples` (the per-iteration values). `n_iterations` records the count. The raw per-iteration cells stay in `results`.

## Bottleneck classes and levers

Each cell gets one governing class, decided in this order:

| Class | Fires when | Lever |
|-------|------------|-------|
| `kv_capacity` | server KV-cache usage ≥ 95 percent, or any preemptions during the cell | add VRAM, reduce context/concurrency, or enable FP8 KV |
| `queue` | requests waiting on the server, or p99/p50 latency > 5 | raise max_num_seqs or add a replica |
| `interconnect` | TP > 1, throughput saturating with concurrency, and concurrency below the critical batch | reduce TP degree or improve interconnect (NVLink) |
| `prefill_compute` | prefill is ≥ 50 percent of request time | increase compute/FLOPs (faster GPU, FP8, or shard the prefill) |
| `decode_compute` | decode-dominated and above the critical batch | already compute-bound; add GPUs/replicas for more throughput |
| `decode_kv_bandwidth` | decode-dominated, below the critical batch, KV traffic dominates weights | increase memory bandwidth (or enable FP8 KV cache) |
| `decode_weight_bandwidth` | decode-dominated, below the critical batch, weights dominate | increase memory bandwidth (or enable FP8 KV cache) |
| `unknown` | no model profile or GPU spec, or no streaming split | collect streaming pp/tg metrics and a known model profile |

Confidence comes from the latency coefficient of variation (under 0.10 high, under 0.30 medium) and is lowered when the analytic and empirical critical batch disagree. The headline verdict is the highest-confidence cell, ties broken toward the highest concurrency.

### GPU spec source

Roofline peaks come from one of two places, chosen by `--gpu-spec`:

- **Derived from the device (NVML).** On an NVIDIA host the suite reads CUDA core count, max SM clock, memory bus width, max memory clock and compute capability and computes: bandwidth = bus width / 8 × memory clock × 2; dense FP16/BF16 peak = SMs × boost clock × FLOP-per-SM-per-clock (2048 on A100, 1024 on professional Ampere/Ada/Blackwell, 512 on GeForce because of the FP32-accumulate half rate, 4096 on Hopper, 8192 on B200); FP8 = 2× on Ada and newer; FP4 = 4× on Blackwell. This is exact for the installed part, including Max-Q and power-limited variants, and is the only source of the FP4 ceiling that NVFP4 / MXFP4 models need.
- **The curated table** (`data/gpu_specs.json`), used on client machines without a driver and as a cross-check.

In `auto` mode both are combined per field: when the table and the device agree within 15 percent the table's datasheet value is kept and marked `table (nvml-confirmed)` (Hopper datasheets use a lower clock basis than NVML's max boost); when they disagree the device wins, the source is reported as `nvml (table overridden)` and the discrepancy is printed and stored in `system_info.gpu_spec._notes`. The resolved spec, the device facts (SM count, clocks, bus width, power limit) and the source are stored under `metadata.system_info.gpu_spec`.

The prefill compute ceiling is chosen by the served weight format: FP4 peak for NVFP4/MXFP4, FP8 peak for FP8, BF16 otherwise (INT4 weight-only quants dequantize to 16-bit for the matmul).

MBU = decode bytes moved per second / peak HBM bandwidth, where bytes per token = active parameters × bytes per parameter + KV bytes per token × context. Reported for single-user cells only. MFU = prefill tokens per second × 2 × active parameters / peak FLOPs for the served dtype (FP8 peak when the quant is FP8). The critical batch B* = peak FLOPs / (HBM bandwidth × bytes per parameter) / 2.

## Score

0 to 10,000, weighted: throughput 30 percent, latency 25, efficiency 20, energy 15, consistency 10.

- Throughput: peak tok/s against the GPU's reference (2000 tok/s when the GPU is unknown).
- Latency: best average latency against a 10 s ceiling.
- Efficiency: best tok/s per user against 200.
- Energy: tokens per watt against the GPU's reference (10 when unknown); 5000 when there is no power telemetry.
- Consistency: latency coefficient of variation against a ceiling of 1.0; 5000 when there is only one request.

Grades: S ≥ 9000, A ≥ 7500, B ≥ 6000, C ≥ 4500, D ≥ 3000, F below. When the GPU matches a baseline the score also reports `gpu_match` and `gpu_percentile` (percent of the reference throughput achieved, capped at 100).

## Statistics

- `confidence_interval`: Student-t interval on the mean. Used for the `*_ci_*` fields.
- `bootstrap_ci`: BCa bootstrap (SciPy), 10,000 resamples, seeded. Used by head-to-head and quant comparison.
- `compare_groups`: Welch's t-test, Mann-Whitney U, Cohen's d, Cliff's delta. `real_difference` requires p < 0.05 and |Cliff's delta| ≥ 0.33.
- `holm_adjusted_p`: Holm-Bonferroni across every comparison in a family.
- `wilson_interval`: for proportions (probe pass rate).
- Degenerate inputs (empty, single value, zero variance) return point estimates and p = 1, never NaN.

## Head-to-head

`--vs URL2` replays the same matrix against the second endpoint, matches cells by (context, concurrency, prompt type) and compares six metrics per cell:

| Metric | Samples used | Direction |
|--------|--------------|-----------|
| `tokens_per_second` | cross-iteration `tokens_per_second_samples` only | higher wins |
| `request_tps` | per-request `request_tps_samples` | higher wins |
| `ttft_estimate` | `ttft_estimate_samples`, else `ttft_samples` | lower wins |
| `prefill_tps` | `prefill_tps_samples` | higher wins |
| `decode_tps` | `decode_tps_samples` | higher wins |
| `avg_latency` | `avg_latency_samples`, else `latency_samples` | lower wins |

A side wins a cell/metric only when all of these hold: `compare_groups` reports a real difference, the Holm-adjusted p is below 0.05, the BCa intervals of the two sample means do not overlap, and the sample means and the cell-level point estimates agree on which side is ahead. Otherwise it is a tie with `tie_reason` = `not_significant` or `direction_conflict`. Each verdict carries `mean_a/mean_b` (sample means), `point_a/point_b` (cell values), `n_a/n_b`, both intervals, the effect sizes and both p-values.

Aggregate `tokens_per_second` has one value per cell per run, so without `--iterations` it is always a tie; `request_tps` is the per-request view and is reported separately because an endpoint that fails requests can have a higher per-request rate and a lower aggregate.

## Quant / model comparison

`--compare-quants a.json b.json [...]` compares every pair of runs on `tokens_per_second` (matched cells), `peak_mem` (`max_mem_used`), `mbu` (from each run's bottleneck verdicts) and `quality` (the run's probe score). Holm correction spans all metrics and pairs; the winner rule is the same as head-to-head minus the direction guard. Output: a ranking by throughput, a verdict per metric and pair, `quant_compare_*.json` and `quant_compare_*.png`.

## Custom prompts

`--prompts-file` takes JSONL with a `"prompt"` key per line. The file is validated and parsed once before the server is contacted; an empty or malformed file exits with status 2.

- The prompt-type axis collapses to a single `custom` type (`--prompt-type` is ignored with a warning).
- Request *i* in every burst, sustained or traffic run gets prompt `i mod N`, so concurrency 8 against a 3-prompt file sends prompts 0,1,2,0,1,2,0,1.
- `avg_prompt_tokens` is the tokenizer-counted mean of the prompts actually sent (counts are memoised per file and tokenizer). The `context_length` of a cell is only the matrix label.
- Cells record `prompt_source="file"`, `prompts_file` and `num_custom_prompts`.

## Quality modes

| Mode | Requires | Output keys |
|------|----------|-------------|
| `probe` | chat completions | `score` (0-100), `passed`, `total`, `overall_ci_low/high` (Wilson 95 percent), per-category breakdown, per-probe results |
| `perplexity` | vLLM with `prompt_logprobs` | `perplexity`, `n_samples`, `n_tokens` |
| `kl` | top-5 logprobs on both endpoints, `--quality-ref` | `mean_kl`, `top1_agreement`, `n_positions`, `tokenizer_mismatch` |

Status is `ok`, `skipped` (with `reason`) or `error`. Quality never feeds into the performance score.

## Regression detection

`--compare FILE` matches cells to the previous run by (context, concurrency, prompt type) and reports changes in `tokens_per_second`, `throughput_per_user` (higher is better), `avg_latency`, `ttft_estimate` (lower is better). Severity is `minor` from 5 percent and `major` from 15 percent. This check is threshold-based and does not use confidence intervals; the GitHub Action can fail on `major`.

## AI analyst report

The report pipeline is: build a facts bundle from the run → send the bundle and a fixed system prompt to the provider → verify every number in the returned text against the bundle (1 percent tolerance, unit-aware) → strike unsupported numbers. More than three unsupported numbers, a provider error, or a malformed response falls back to a deterministic template with the same five sections (executive summary, bottleneck analysis, application fitness, recommendations, caveats). The bundle's SHA-256 is recorded in `report_*.md`.

Providers: `local` posts to the benchmarked server's chat endpoint; `openai` to any OpenAI-compatible `--report-llm-url`; `claude` uses the Anthropic SDK (`pip install vllm-benchmark-suite[report-claude]`, `ANTHROPIC_API_KEY` in the environment).

## GitHub Action

Inputs: `vllm-url` (default `http://localhost:8000`), `preset` (`quick`), `context-lengths`, `concurrency`, `output-tokens` (`500`), `prompt-type` (`classic`), `compare-with`, `fail-on-regression` (`false`), `python-version` (`3.12`). Outputs: `score`, `grade`, `peak-throughput`, `results-json`. The action installs the latest release from PyPI, so pin the action to a tag that matches the version you expect.
