# vLLM Benchmark Suite

Point it at a vLLM or SGLang server, get tok/s, TTFT, latency percentiles, a roofline bottleneck verdict and a Reddit-ready summary. Compare quants, compare engines, and know whether the difference is real or noise.

[![PyPI](https://img.shields.io/pypi/v/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![Python](https://img.shields.io/pypi/pyversions/vllm-benchmark-suite)](https://pypi.org/project/vllm-benchmark-suite/)
[![CI](https://github.com/notaDestroyer/vllm-benchmark-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/notaDestroyer/vllm-benchmark-suite/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install vllm-benchmark-suite
vllm-bench --url http://localhost:8000 --quick -y
```

Five minutes later you have a score, a bottleneck diagnosis, charts, an HTML report and a JSON file with every number. `llm-bench` is the same command under a shorter name.

---

## Why this and not a script with `time.time()`

Most "my 3090 does X tok/s" posts measure one thing, once, with one concurrent request. This tool:

- **Hits the server properly.** True async concurrency via `aiohttp`, real Time-to-First-Token from the SSE stream, and prefill vs decode throughput measured separately instead of guessed from a ratio.
- **Counts tokens with the model's tokenizer**, not `len(text) // 4`. Use `--tokenizer` for gated or locally served models.
- **Knows what it is benchmarking.** Dense or MoE, active vs total parameters, GQA/MQA, KV-cache bytes per token. That feeds a roofline model that tells you whether each load point is bandwidth-bound, compute-bound, queueing, or out of KV cache, and what to change.
- **Knows your GPU from the GPU.** On the host it derives bandwidth and dense BF16/FP8/FP4 peaks from the card's own SM count, clocks and bus width via NVML, so a 300 W Max-Q and a 600 W PRO 6000 get their real ceilings, NVFP4 models get an FP4 ceiling, and the built-in table is only a cross-check.
- **Tells you when a difference is noise.** `--iterations` gives confidence intervals; comparisons use Welch, Mann-Whitney, effect sizes and Holm correction. A "win" is only declared when it is statistically and practically significant.
- **Makes the post for you.** `--share` writes a Markdown table you can paste into a thread, plus a result-card PNG.

---

## Get started

### 1. Serve something

```bash
# vLLM
vllm serve Qwen/Qwen3-7B-Instruct-AWQ --max-model-len 32768
# or SGLang (add --enable-metrics so the KV-cache / queue signals are available)
python -m sglang.launch_server --model-path Qwen/Qwen3-7B-Instruct-AWQ --enable-metrics
```

### 2. Run a preset

```bash
vllm-bench --url http://localhost:8000 --quick -y        # ~5 min:  32K, 1 and 4 users
vllm-bench --url http://localhost:8000 --standard -y     # ~30 min: 32K/64K/128K, 1-16 users
vllm-bench --url http://localhost:8000 --thorough -y     # ~2 h:    up to 512K, 1-32 users, all prompt types
```

| Preset | Context lengths | Concurrency | Prompt types | Output tokens |
|--------|-----------------|-------------|--------------|---------------|
| `--quick` | 32K | 1, 4 | classic | 150 |
| `--standard` | 32K, 64K, 128K | 1, 4, 8, 16 | classic, deterministic | 500 |
| `--thorough` | 32K to 512K | 1, 4, 8, 16, 32 | all four | 500 |

Or set the matrix yourself. Context lengths must fit the server's `--max-model-len`:

```bash
vllm-bench --url http://localhost:8000 --context-lengths 4k,8k,16k --concurrency 1,2,4,8 --output-tokens 256 -y
```

The backend (vLLM or SGLang) is auto-detected. Force it with `--backend vllm|sglang`.

### 3. Read the output

Trimmed example of what you see at the end of a run. Numbers are illustrative, not a measurement.

```
OK 1/4 32K x 1u x classic: 48.3 tok/s, 10.41s
OK 2/4 32K x 4u x classic: 171.9 tok/s, 11.63s
...
Bottleneck: decode_weight_bandwidth (confidence high) -> increase memory bandwidth (or enable FP8 KV cache)
Config tips:
  - GQA attention keeps the KV cache small - you can push longer context and higher concurrency before hitting KV limits.

╭──────────────────── vLLM Benchmark Score ────────────────────╮
│   vLLM Benchmark Score: 6,420 / 10,000  (Grade: B)           │
│   Throughput  (30%) ███████████████░░░░░░░░░░░░░░░ 4,980     │
│   Latency     (25%) ████████████████████████░░░░░░ 8,110     │
│   Efficiency  (20%) ██████████░░░░░░░░░░░░░░░░░░░░ 3,350     │
│   Energy      (15%) ██████████████████░░░░░░░░░░░░ 6,000     │
│   Consistency (10%) ████████████████████████████░░ 9,400     │
╰──────────────────────────────────────────────────────────────╯

╭──────────────────── Bottleneck Analysis ─────────────────────╮
│ Governing: decode_weight_bandwidth (confidence high)         │
│   Cell (ctx/users)   Primary                   MBU   MFU     │
│   32K/1              decode_weight_bandwidth   61%   N/A     │
│   32K/4              decode_weight_bandwidth   N/A   12%     │
│   32K/16             queue                     N/A   14%     │
╰──────────────────────────────────────────────────────────────╯
```

How to read it:

- **The `OK` lines** are one per cell: context length, concurrent users, prompt type, aggregate tok/s and average request latency.
- **Bottleneck** is the governing limit at the most loaded high-confidence cell, with the lever to pull. The classes are `decode_weight_bandwidth`, `decode_kv_bandwidth`, `decode_compute`, `prefill_compute`, `queue`, `kv_capacity`, `interconnect`, or `unknown` when the model or GPU could not be identified.
- **MBU** is the fraction of the GPU's HBM bandwidth you are using during single-user decode. On a consumer card at batch 1 this is the number that matters: 60 to 80 percent is about as good as it gets. **MFU** is the fraction of peak FLOPs during prefill; it is a ±15 percent estimate, not a billing figure.
- **The score** is a 0 to 10,000 composite against a per-GPU baseline (H100, A100, L40S, RTX 4090/5090, RTX 6000 Ada, RTX PRO 6000 Blackwell). Grades: S at 9000, A at 7500, B at 6000, C at 4500, D at 3000, F below. Energy scores default to the midpoint when there is no nvidia-smi telemetry, so a CPU-only client machine gets a neutral energy score, not a penalty.

Below that comes the **Capacity & Ceilings** panel, which is where "how do I get the most out of this model on this card" is answered with numbers instead of tips:

```
╭──────────────────────────── Capacity & Ceilings ─────────────────────────────╮
│ Weights         8.0 GB (FP8, 1.0 B/param)                                    │
│ KV bytes/token  131,072                                                      │
│ KV capacity     441,344 tokens (server)                                      │
│ GPU memory      102.6 GB x 1 @ 90% utilisation                               │
│ Max users by context  32K: 13, 128K: 3                                       │
│ Over capacity   32K x 16u, 128K x 4u, 128K x 8u                              │
│                                                                              │
│  Cell   Decode ceiling  Measured  Attained  Bound      Prefill ceiling  MFU  │
│  32K/1      147 tok/s    112 tok/s    76%   bandwidth     62,740 tok/s  31%  │
│  32K/4      530 tok/s    402 tok/s    76%   bandwidth     62,740 tok/s  29%  │
│  32K/16   1,560 tok/s    980 tok/s    63%   bandwidth     62,740 tok/s  27%  │
│                                                                              │
│ Operating point: Gains flatten at 8 users (32K), well below the hardware     │
│   critical batch (~280): the limit is the server (KV capacity, queueing).    │
│ Power: Power-limited: ran at 74% of max SM clock, power-capped 91% of the    │
│   time at 300 W. A higher power limit (or the 600 W variant) is the lever.   │
│ What-if: NVFP4 would raise the single-stream ceiling to 254 tok/s and KV     │
│   room to ~650,000 tokens (needs Blackwell; FP4 tensor cores).               │
│ Suggested serve command (KV capacity 441,344 tokens (server); ...):          │
│ vllm serve meta-llama/Llama-3-8B --max-model-len 128500 ...                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

- **KV capacity** is the number of tokens the server's KV cache holds. On vLLM it is read from the server itself (`cache_config_info`), on SGLang from `max_total_num_tokens`; elsewhere it is estimated from GPU memory. It sets **max users by context** and flags matrix cells that cannot fit before you wait for them to fail. The same check runs *before* the benchmark starts.
- **Decode ceiling** is the roofline limit for this model on this card at that concurrency: `N × bandwidth / (weights + N × KV bytes × context)`, capped by compute. **Attained** is what you got as a fraction of it. 60 to 80 percent at batch 1 is normal; a gap that grows with concurrency points at the server.
- **Operating point** compares the measured knee of the throughput curve with the hardware's critical batch, so it can say whether the GPU ran out of compute or the server ran out of something else.
- **Power** reads the sustained SM clock, the enforced power limit and the throttle reasons while the cells ran. A 300 W Max-Q card is told apart from a 600 W one here, not guessed from the name.
- **What-if** projects weights, KV room and the single-stream ceiling for BF16, FP8, INT4 and NVFP4 (and FP8 KV cache) on *this* card, marking formats the card cannot run.
- **Suggested serve command** is sized from those numbers: `--max-model-len` from the KV budget, `--max-num-seqs` from the users that fit at the largest context, never below the measured sweet spot.

The formulas and their sources are in the [reference](docs/reference.md#capacity-and-ceilings). The built-in roofline table covers 49 cards from Ampere to Blackwell (B200, RTX PRO 6000/5000/4500 Blackwell, RTX 5060 to 5090, the Hopper and Ada datacenter parts, RTX 30/40, and more); the full list with their numbers is in [docs/gpus.md](docs/gpus.md). On the GPU host the spec is derived from the device itself, so an unlisted card still gets a correct roofline.

Everything also lands in `./outputs/`: `benchmark_*.json` (all numbers, plus `model_profile`, `bottlenecks`, `advisory` with its `capacity` section, `quality`), a CSV, PNG charts including the roofline, a context × concurrency bottleneck map and the capacity figure (measured vs ceiling by concurrency, context slope, what-if by format), and an interactive HTML report.

---

## Things that matter

### "Is AWQ or GPTQ or FP8 faster on my card?"

Run each quant, then compare the JSON files. The comparison is Holm-corrected across every metric and pair, and declares a tie unless the difference is real.

```bash
vllm-bench --url http://localhost:8000 --standard --iterations 3 -y --output-dir out/awq
vllm-bench --url http://localhost:8000 --standard --iterations 3 -y --output-dir out/fp8
vllm-bench --compare-quants out/awq/benchmark_*.json out/fp8/benchmark_*.json
```

Add `--quality probe` to each run and the comparison includes a correctness score, so you can see what the quant cost you. `--iterations 3` is what makes the statistics meaningful; a single run per quant can only ever tie.

### "vLLM or SGLang for this model?"

Serve the same model on both, then run the matrix once and let the tool replay it against the second endpoint:

```bash
vllm-bench --url http://localhost:8000 --standard -y --vs http://localhost:30000
```

You get a per-cell, per-metric verdict in `head_to_head_*.json`. Per-request throughput is compared as its own metric and never stands in for aggregate throughput, because an engine that drops half its requests can look faster per request while being slower overall. The exact rules are in the [reference](docs/reference.md#head-to-head).

### "What's the real-world number, not the burst number?"

Burst mode fires all N requests at once and measures peak. For "what does this feel like with people actually using it":

```bash
vllm-bench --url http://localhost:8000 --rps 2 --duration 120 -y        # steady 2 req/s for two minutes
vllm-bench --url http://localhost:8000 --traffic poisson --target-rps 2  # random arrivals
vllm-bench --url http://localhost:8000 --traffic multiturn --turns 5     # growing chat contexts
```

### "Can I use my own prompts?"

Yes, and they are used everywhere: burst, sustained, and traffic modes.

```bash
vllm-bench --url http://localhost:8000 --prompts-file my_prompts.jsonl --concurrency 1,4,8 -y
```

One JSON object per line with a `"prompt"` key. Prompts are cycled across requests, cells are tagged `prompt_type=custom`, and the real prompt length shows up as `avg_prompt_tokens`. The built-in strategies are `classic` (realistic, high prefix-cache hits), `deterministic` (identical prefixes, best-case cache), `madlib` (moderate misses) and `random` (worst case). `--prompt-type all` sweeps them.

### "How much context can I actually fit?"

The Capacity panel answers it directly: KV capacity in tokens (read from the server when it publishes it), max users at each context, and which matrix cells are over capacity. The bottleneck map shows where cells flip to `kv_capacity` (the server's own KV-cache usage hit 95 percent, or it started preempting). Run `--bottleneck-sweep` to measure the critical batch size on your hardware instead of trusting the analytic estimate.

### "Did my config change make it worse?"

```bash
vllm-bench --url http://localhost:8000 --standard -y --compare outputs/benchmark_before.json
```

Matches cells by context, concurrency and prompt type and flags changes beyond fixed thresholds (5 percent minor, 15 percent major). This check is threshold-based; use `--iterations` with `--compare-quants` when you need statistical significance.

### "I want to post this"

```bash
vllm-bench --url http://localhost:8000 --standard -y --share
```

Writes `share_*.md`, a table with GPU, model, quant, peak tok/s, prefill/decode split, best TTFT, MBU, bottleneck and score, plus a `<details>` block with the full matrix, and renders `result_card.png`. Paste the Markdown, attach the card.

### "Can an LLM write the analysis for me?"

```bash
vllm-bench --url http://localhost:8000 --standard -y --ai-report                      # the benchmarked server writes it
vllm-bench ... --ai-report --report-provider openai --report-llm-url http://llm:8000   # any OpenAI-compatible endpoint
vllm-bench ... --ai-report --report-provider claude                                    # needs pip install vllm-benchmark-suite[report-claude]
```

The LLM only writes prose. Every number is computed by the tool, and a verifier strikes out any figure in the text that is not in the run's data. If generation fails or too many numbers are unsupported, you get a deterministic template report instead. The report goes to `report_*.md` and into the HTML.

---

## Quality: did the quant break the model?

Performance and quality are reported separately and never mixed into the score.

```bash
vllm-bench ... --quality probe                                   # short correctness probes -> score /100 with a Wilson CI
vllm-bench ... --quality perplexity                              # vLLM only: needs prompt logprobs
vllm-bench ... --quality kl --quality-ref http://fp16-server:8000  # KL divergence and top-1 agreement vs a reference
```

`perplexity` and `kl` need token-level logprobs from the server and are skipped with a reason when unavailable. The probe set is ten short arithmetic, factual, format and instruction checks: enough to catch a broken quant, not a replacement for a real eval.

---

## Cost and energy

Cost per million tokens is reported per cell when a `$/hr` rate is known. Pass `--cost 0.74` or let it auto-detect from the GPU name for H100, A100, L40S, L4, A10G, T4, RTX 4090/5090, RTX 6000 Ada and RTX PRO 6000 Blackwell. The rate is multiplied by the GPUs the server actually occupies (tensor × pipeline parallel), not the whole box. Tokens per watt comes from nvidia-smi power draw when the benchmark runs on the GPU host.

GPU names are matched by token, so `NVIDIA A100-SXM4-80GB`, `NVIDIA H100 80GB HBM3` and `4x NVIDIA H100 80GB HBM3` all resolve, and a 3090 or A6000 is never mistaken for a 4090. Unknown cards get no baseline and no price; the run still works.

---

## Multi-GPU and remote clients

- Run the benchmark on the GPU host to get nvidia-smi telemetry, energy and temperature diagnostics. Run it anywhere else and those sections are simply omitted; a CPU-only client prints one warning, not one per poll.
- On multi-GPU hosts the GPU name collapses to `Nx <name>`, `gpu_count` is recorded and VRAM is summed.
- Server-side signals (KV-cache usage, waiting and running requests, preemptions) come from the server's `/metrics` endpoint and are sampled during every cell. vLLM exposes it by default; SGLang needs `--enable-metrics`.

---

## Automation

**GitHub Action.** The repo ships a composite action that installs the suite, runs a preset against a server the runner can reach, uploads the results directory, exposes `score`, `grade`, `peak-throughput` and `results-json` as outputs, and can fail the job on major regressions against a stored baseline.

```yaml
- uses: notaDestroyer/vllm-benchmark-suite@v4.2.2
  id: bench
  with:
    vllm-url: http://localhost:8000
    preset: quick
    compare-with: baseline/benchmark.json
    fail-on-regression: "true"
- run: echo "Score ${{ steps.bench.outputs.score }} (${{ steps.bench.outputs.grade }})"
```

**Grafana.** `grafana/dashboard.json` is a dashboard for a Prometheus data source scraping vLLM's `/metrics` directly: throughput, latency, TTFT, ITL, KV-cache usage, prefix-cache hit rate and GPU telemetry. It watches the server, independent of this tool; import it from Dashboards → New → Import.

**Leaderboard.** Every run writes a standardised `result_*.json`. `vllm-bench --leaderboard outputs/` turns a directory of them into `LEADERBOARD.md`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `Error loading model tokenizer! Using google/gemma3-4b-it instead` | The served model name is not a public HF repo (gated, local path, custom name). Pass `--tokenizer org/repo-or-local-path` or set `VLLM_BENCH_TOKENIZER`. Token counts are still correct from the server's `usage` field; this only affects prompt generation length. |
| Bottleneck is `unknown`, MBU/MFU are `N/A` | Run on the GPU host so the spec can be derived from the device (`--gpu-spec auto`, needs the NVIDIA driver), or the model could not be profiled: check `model_profile.source` and `system_info.gpu_spec._source` in the JSON. The fallback table lives in `src/vllm_benchmark/data/gpu_specs.json`. |
| MFU looks 2x too high or low | Check `system_info.gpu_spec._source`. GeForce cards run FP16 tensor math at half rate with FP32 accumulate; professional cards do not. The device-derived spec handles this; the table entry may not. `--gpu-spec nvml` forces the derivation. |
| `kv_capacity` / `queue` never appear, cache hit rate is empty | `/metrics` is not reachable. On SGLang start the server with `--enable-metrics`. Behind a proxy, make sure `/metrics` is forwarded. |
| Warmup fails, or the first cell times out | Cold model and a large first prompt. `--no-warmup` skips the warmup request; `--context-lengths` smaller than `--max-model-len` avoids the 400 from the server. |
| All requests fail with HTTP 400 at a context length | The prompt exceeds the server's `--max-model-len`. Lower `--context-lengths`. |
| Every comparison is a tie | Single runs only have point estimates. Use `--iterations 3` or more on both sides. |
| `nvidia-smi not found` | You are running on a client machine. Expected; GPU sections are omitted. |

---

## What it does not do

- It measures the server from the outside. Internal scheduler timing comes only from what `/metrics` exposes.
- MFU is an approximation from `2 × active parameters` FLOPs per token. Treat it as a guide.
- The quality probes are a smoke test, not an eval harness. Use lm-eval or similar for real quality numbers.
- There is no GPU support beyond NVIDIA for telemetry. AMD and Apple hosts run fine; they just get no power, temperature or utilisation data.
- Prices and score baselines are hand-maintained tables. If your card is missing, open an issue or a PR against `src/vllm_benchmark/config.py` and `data/gpu_specs.json`.

---

## Reference

The full CLI reference, every metric in the JSON, output file list, the statistics used, the head-to-head rules and the custom-prompt semantics are in [docs/reference.md](docs/reference.md). Changes per release are in [CHANGELOG.md](CHANGELOG.md).

---

## Install

```bash
pip install vllm-benchmark-suite                 # or: uv pip install vllm-benchmark-suite
pip install "vllm-benchmark-suite[report-claude]"  # adds the Claude provider for --ai-report
```

From source:

```bash
git clone https://github.com/notaDestroyer/vllm-benchmark-suite.git
cd vllm-benchmark-suite
pip install -e ".[dev]"
```

Python 3.10 to 3.13. A running vLLM or SGLang server. NVIDIA GPU optional, only needed for telemetry.

---

## Contributing

```bash
pip install -e ".[dev]"
pytest                      # ~650 tests, no GPU or server needed (the e2e test uses an in-process mock)
ruff check src/ tests/
```

CI runs the suite on Python 3.10 through 3.13 and enforces 70 percent coverage. Open an issue before large changes; new GPUs and models in the data tables are welcome as small PRs.

## License

MIT
