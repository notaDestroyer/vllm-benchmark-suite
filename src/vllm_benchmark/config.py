"""Benchmark configuration management."""

from dataclasses import dataclass, field
from typing import Optional

# Preset benchmark profiles
PRESETS = {
    "quick": {
        "context_lengths": [32_000],
        "concurrency_levels": [1, 4],
        "output_tokens": 150,
        "prompt_types": ["classic"],
        "description": "Quick smoke test (~5 min)",
    },
    "standard": {
        "context_lengths": [32_000, 64_000, 128_000],
        "concurrency_levels": [1, 4, 8, 16],
        "output_tokens": 500,
        "prompt_types": ["classic", "deterministic"],
        "description": "Standard benchmark (~30 min)",
    },
    "thorough": {
        "context_lengths": [32_000, 64_000, 128_000, 256_000, 512_000],
        "concurrency_levels": [1, 4, 8, 16, 32],
        "output_tokens": 500,
        "prompt_types": ["classic", "deterministic", "madlib", "random"],
        "description": "Comprehensive benchmark (~2 hours)",
    },
}

# Default tokenizer for prompt generation
DEFAULT_TOKENIZER = "google/gemma3-4b-it"

# GPU reference baselines for scoring (tokens/sec at 32K context, 4 users)
GPU_BASELINES = {
    "H100": {"throughput_ref": 2200, "ttft_ref_ms": 80, "tpw_ref": 12.0},
    "A100-80GB": {"throughput_ref": 1400, "ttft_ref_ms": 120, "tpw_ref": 8.0},
    "A100-40GB": {"throughput_ref": 1000, "ttft_ref_ms": 150, "tpw_ref": 7.0},
    "L40S": {"throughput_ref": 900, "ttft_ref_ms": 140, "tpw_ref": 6.5},
    "RTX 6000 Ada": {"throughput_ref": 800, "ttft_ref_ms": 160, "tpw_ref": 5.5},
    "RTX 4090": {"throughput_ref": 750, "ttft_ref_ms": 170, "tpw_ref": 5.0},
    "RTX 5090": {"throughput_ref": 850, "ttft_ref_ms": 150, "tpw_ref": 5.5},
    "RTX PRO 6000": {"throughput_ref": 900, "ttft_ref_ms": 140, "tpw_ref": 6.0},
}


def parse_context_lengths(text: str) -> list[int]:
    """Parse context lengths from a string like '32k,64k,128k' or '32000,64000'."""
    lengths = []
    for part in text.split(","):
        part = part.strip().lower()
        if part.endswith("k"):
            lengths.append(int(float(part[:-1]) * 1000))
        elif part.endswith("m"):
            lengths.append(int(float(part[:-1]) * 1_000_000))
        else:
            lengths.append(int(part))
    return sorted(lengths)


def parse_concurrency(text: str) -> list[int]:
    """Parse concurrency levels from a string like '1,4,8,16'."""
    return sorted(int(x.strip()) for x in text.split(","))


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    # Connection
    api_url: str = "http://localhost:8000"
    model_name: Optional[str] = None  # Auto-detected if None

    # Test matrix
    context_lengths: list[int] = field(default_factory=lambda: [32_000, 64_000, 128_000])
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    output_tokens: int = 500
    prompt_types: list[str] = field(default_factory=lambda: ["classic"])

    # Behavior
    warmup: bool = True
    pause_between_tests: int = 5
    request_timeout: int = 900
    gpu_poll_interval: float = 0.1
    streaming: bool = True  # Use streaming for true TTFT measurement

    # Output
    output_dir: str = "./outputs"
    generate_html: bool = True
    generate_charts: bool = True

    # Comparison
    compare_file: Optional[str] = None  # Path to previous results JSON
    prompts_file: Optional[str] = None  # Path to custom prompts JSONL

    @property
    def api_endpoint(self) -> str:
        return f"{self.api_url}/v1/chat/completions"

    @property
    def models_endpoint(self) -> str:
        return f"{self.api_url}/v1/models"

    @property
    def health_endpoint(self) -> str:
        return f"{self.api_url}/health"

    @property
    def version_endpoint(self) -> str:
        return f"{self.api_url}/version"

    @property
    def metrics_endpoint(self) -> str:
        return f"{self.api_url}/metrics"

    @property
    def total_tests(self) -> int:
        return len(self.context_lengths) * len(self.concurrency_levels) * len(self.prompt_types)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "BenchmarkConfig":
        """Create config from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from: {list(PRESETS.keys())}")
        preset = PRESETS[preset_name]
        return cls(
            context_lengths=preset["context_lengths"],
            concurrency_levels=preset["concurrency_levels"],
            output_tokens=preset["output_tokens"],
            prompt_types=preset["prompt_types"],
            **overrides,
        )
