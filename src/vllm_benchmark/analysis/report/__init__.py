"""AI analyst report: turn already-computed facts into prose.

The language model is a writer, not a calculator — every number it may use
is computed by earlier PRs and assembled into a facts bundle here.  A
post-generation numeric verifier rejects any unsupported figure, and a
deterministic template is the fallback (and the no-LLM output).
"""

from vllm_benchmark.analysis.report.analyst import (
    SYSTEM_PROMPT,
    AnalystReport,
    deterministic_report,
    generate_report,
)
from vllm_benchmark.analysis.report.bundle import (
    allowed_numbers,
    build_bundle,
    bundle_sha256,
    sanitize_text,
)
from vllm_benchmark.analysis.report.providers import (
    ClaudeProvider,
    LocalProvider,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    get_provider,
)
from vllm_benchmark.analysis.report.verify import extract_numbers, verify_report

__all__ = [
    "SYSTEM_PROMPT",
    "AnalystReport",
    "deterministic_report",
    "generate_report",
    "build_bundle",
    "allowed_numbers",
    "bundle_sha256",
    "sanitize_text",
    "Provider",
    "ProviderError",
    "LocalProvider",
    "OpenAICompatProvider",
    "ClaudeProvider",
    "get_provider",
    "extract_numbers",
    "verify_report",
]
