"""Model warmup for vLLM benchmark runs.

Author: amit
License: MIT
"""

from __future__ import annotations

import time

import requests
from rich.console import Console

from vllm_benchmark.config import BenchmarkConfig

console = Console()


def warmup_model(config: BenchmarkConfig, model_name: str, output_tokens: int = 100) -> bool:
    """Execute warmup inference to initialize GPU kernels and caches."""
    console.print("\n[yellow]Warming up model (1K context, single user)...[/yellow]")

    from vllm_benchmark.core.prompts import generate_prompt
    prompt = generate_prompt(1000)
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.7,
    }

    try:
        with console.status("[bold yellow]Executing warmup inference...", spinner="dots"):
            start = time.time()
            response = requests.post(config.api_endpoint, json=data, timeout=config.request_timeout)
            duration = time.time() - start

            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                console.print(
                    f"[green]OK[/green] Warmup complete: {duration:.2f}s, "
                    f"{completion_tokens} tokens generated"
                )
                return True
            else:
                console.print(f"[red]FAIL[/red] Warmup failed: HTTP {response.status_code}")
                return False
    except Exception as e:
        console.print(f"[red]FAIL[/red] Warmup error: {str(e)}")
        return False
