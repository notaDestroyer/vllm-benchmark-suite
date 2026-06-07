"""Structured-output / function-calling workload.

Sends chat requests that constrain the model to emit JSON (via
``response_format``) or to call a tool (via ``tools``), then scores how
reliably the model produced *valid, schema-conforming* output.

The scoring core, :func:`score_structured_response`, is a pure function
with no network dependency so it can be unit-tested directly against
valid JSON, malformed JSON, schema-violating JSON, correct/incorrect tool
calls and truncated streaming fragments.

Metrics surfaced by :meth:`StructuredWorkload.summarize`:

* ``schema_adherence_rate`` — fraction of schema probes whose response is
  valid JSON *and* validates against the probe's JSON schema,
* ``tool_call_correctness`` — fraction of tool probes that selected the
  expected tool with all required arguments present,
* TTFT and p99/p50 latency-consistency derived from per-request timing.

Author: amit
License: MIT
"""

from __future__ import annotations

import importlib.resources
import json
import re
import time
from functools import lru_cache
from statistics import mean
from typing import TYPE_CHECKING, Any, Optional

import aiohttp

from vllm_benchmark.core.async_engine import calculate_percentiles
from vllm_benchmark.core.workloads.base import Workload

try:  # jsonschema is a hard dependency for schema validation
    import jsonschema
    from jsonschema import Draft7Validator
except Exception:  # pragma: no cover - import guard only
    jsonschema = None  # type: ignore[assignment]
    Draft7Validator = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm_benchmark.config import BenchmarkConfig
    from vllm_benchmark.core.backends.base import ServerInfo


# ---------------------------------------------------------------------------
# Probe-set loader (packaged JSON)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_structured_probes() -> dict:
    """Load the bundled structured-output probe set.

    Returns:
        A dict with ``schema_probes`` and ``tool_probes`` lists (the
        ``_README`` key is stripped).
    """
    resource = importlib.resources.files("vllm_benchmark.data.structured").joinpath(
        "probes.json"
    )
    with resource.open("r", encoding="utf-8") as f:
        data = dict(json.load(f))
    data.pop("_README", None)
    return data


# ---------------------------------------------------------------------------
# Pure scoring core
# ---------------------------------------------------------------------------

#: Matches a fenced ```json ... ``` block so we can salvage wrapped output.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON object/array from ``text``.

    Tries a direct ``json.loads`` first, then a fenced code block, then the
    first ``{...}`` / ``[...]`` span.  Returns ``None`` when nothing parses
    (e.g. a truncated streaming fragment).
    """
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None

    # 1. Direct parse.
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fenced code block.
    fence = _FENCE_RE.search(candidate)
    if fence:
        try:
            return json.loads(fence.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. First balanced-looking object/array span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = candidate.find(open_ch)
        end = candidate.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _validate_against_schema(obj: Any, schema: Optional[dict]) -> bool:
    """Return whether ``obj`` validates against ``schema`` (Draft 7).

    A missing schema means "any valid JSON is acceptable".  Validation
    errors (including a missing ``jsonschema`` install) yield ``False``
    rather than raising.
    """
    if schema is None:
        return True
    if Draft7Validator is None:  # pragma: no cover - dependency guard
        return False
    try:
        Draft7Validator(schema).validate(obj)
        return True
    except jsonschema.ValidationError:
        return False
    except Exception:
        return False


def score_structured_response(
    text: str,
    schema: Optional[dict],
    expected_tool: Optional[str] = None,
    *,
    tool_calls: Optional[list[dict]] = None,
    required_args: Optional[list[str]] = None,
) -> dict:
    """Score a single structured-output response.

    Pure and network-free.  Handles two cases:

    * **JSON schema** — ``text`` is parsed (direct, fenced or salvaged
      span) and validated against ``schema``.
    * **Tool call** — ``tool_calls`` (OpenAI-style) is inspected for the
      ``expected_tool`` with all ``required_args`` present in its parsed
      arguments.

    Args:
        text: The assistant message content (may be empty for tool calls).
        schema: JSON schema to validate against, or ``None`` to require
            only well-formed JSON.
        expected_tool: When set, score this as a tool-call probe.
        tool_calls: OpenAI-style ``tool_calls`` list from the response.
        required_args: Argument names that must be present in the call.

    Returns:
        A dict with ``valid_json``, ``schema_valid``, ``tool_correct`` and
        an overall ``adherent`` boolean.  Fields that do not apply to the
        probe type are ``None``.
    """
    result: dict = {
        "valid_json": None,
        "schema_valid": None,
        "tool_correct": None,
        "adherent": False,
    }

    if expected_tool is not None:
        result["tool_correct"] = _score_tool_call(
            tool_calls, expected_tool, required_args or []
        )
        result["adherent"] = bool(result["tool_correct"])
        return result

    obj = _extract_json_object(text)
    result["valid_json"] = obj is not None
    if obj is None:
        result["schema_valid"] = False
        result["adherent"] = False
        return result

    schema_ok = _validate_against_schema(obj, schema)
    result["schema_valid"] = schema_ok
    result["adherent"] = bool(schema_ok)
    return result


def _score_tool_call(
    tool_calls: Optional[list[dict]],
    expected_tool: str,
    required_args: list[str],
) -> bool:
    """Return whether ``tool_calls`` correctly invokes ``expected_tool``."""
    if not tool_calls:
        return False
    for call in tool_calls:
        fn = call.get("function", call) if isinstance(call, dict) else {}
        name = fn.get("name")
        if name != expected_tool:
            continue
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except (json.JSONDecodeError, ValueError):
                return False
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if all(a in args for a in required_args):
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# HTTP helper (monkeypatchable for tests)
# ---------------------------------------------------------------------------

async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    body: dict,
    request_timeout: int,
) -> dict:
    """POST a chat-completion probe and measure its wall-clock latency.

    The sole network boundary of the workload; monkeypatched in tests.

    Returns:
        A per-request dict with ``success``, ``duration``, ``content`` and
        ``tool_calls`` (best-effort extracted from the first choice).
    """
    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(endpoint, json=body, timeout=timeout) as response:
            duration = time.perf_counter() - start
            if response.status != 200:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"HTTP {response.status}",
                }
            payload = await response.json()
            message = (payload.get("choices") or [{}])[0].get("message", {})
            return {
                "success": True,
                "duration": duration,
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls"),
            }
    except Exception as exc:  # network errors are recorded, never raised
        return {
            "success": False,
            "duration": time.perf_counter() - start,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class StructuredWorkload(Workload):
    """Structured-output / function-calling fitness workload.

    Runs a small bundled probe set: JSON-schema extraction probes and a
    tool-call probe.  Reports schema adherence, tool-call correctness, and
    TTFT / latency-consistency timing.
    """

    name = "structured"

    def applicable(self, server_info: "ServerInfo") -> bool:
        """Applicable for ``generate`` (or unknown) server tasks."""
        task = getattr(server_info, "task", "unknown")
        return task in ("generate", "unknown")

    async def run(
        self,
        config: "BenchmarkConfig",
        model_name: str,
        *,
        repeats: int = 1,
        **kwargs: Any,
    ) -> list[dict]:
        """Run the bundled schema + tool probes against the chat endpoint.

        Each probe may be repeated ``repeats`` times to get a stable
        adherence estimate.  Returns one cell per probe execution.  Cells
        carry timing plus the scored ``adherent`` / ``schema_valid`` /
        ``tool_correct`` flags.
        """
        probes = load_structured_probes()
        endpoint = f"{config.api_url}/v1/chat/completions"
        results: list[dict] = []

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            for _ in range(max(1, repeats)):
                for probe in probes.get("schema_probes", []):
                    cell = await self._run_schema_probe(
                        session, endpoint, model_name, probe, config.request_timeout
                    )
                    if cell is not None:
                        results.append(cell)
                for probe in probes.get("tool_probes", []):
                    cell = await self._run_tool_probe(
                        session, endpoint, model_name, probe, config.request_timeout
                    )
                    if cell is not None:
                        results.append(cell)
        return results

    async def _run_schema_probe(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        model_name: str,
        probe: dict,
        request_timeout: int,
    ) -> Optional[dict]:
        """Execute one JSON-schema probe and score the response."""
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "max_tokens": 256,
            "temperature": 0.0,
            "response_format": probe.get(
                "response_format", {"type": "json_object"}
            ),
        }
        req = await _post_chat(session, endpoint, body, request_timeout)
        if not req.get("success"):
            return None
        score = score_structured_response(req.get("content", ""), probe.get("schema"))
        return {
            "workload": self.name,
            "probe_id": probe.get("id"),
            "probe_kind": "schema",
            "avg_latency": req.get("duration"),
            "ttft_estimate": req.get("duration"),
            "valid_json": score["valid_json"],
            "schema_valid": score["schema_valid"],
            "adherent": score["adherent"],
        }

    async def _run_tool_probe(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        model_name: str,
        probe: dict,
        request_timeout: int,
    ) -> Optional[dict]:
        """Execute one tool-call probe and score the response."""
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "max_tokens": 256,
            "temperature": 0.0,
            "tools": probe.get("tools", []),
            "tool_choice": "auto",
        }
        req = await _post_chat(session, endpoint, body, request_timeout)
        if not req.get("success"):
            return None
        score = score_structured_response(
            req.get("content", ""),
            None,
            expected_tool=probe.get("expected_tool"),
            tool_calls=req.get("tool_calls"),
            required_args=probe.get("required_args"),
        )
        return {
            "workload": self.name,
            "probe_id": probe.get("id"),
            "probe_kind": "tool",
            "avg_latency": req.get("duration"),
            "ttft_estimate": req.get("duration"),
            "tool_correct": score["tool_correct"],
            "adherent": score["adherent"],
        }

    def summarize(self, results: list[dict]) -> dict:
        """Summarize adherence, tool correctness and timing consistency."""
        if not results:
            return {"workload": self.name, "cells": 0}

        schema_cells = [r for r in results if r.get("probe_kind") == "schema"]
        tool_cells = [r for r in results if r.get("probe_kind") == "tool"]

        schema_adherence = (
            mean(1.0 if c.get("adherent") else 0.0 for c in schema_cells)
            if schema_cells
            else None
        )
        tool_correctness = (
            mean(1.0 if c.get("adherent") else 0.0 for c in tool_cells)
            if tool_cells
            else None
        )

        latencies = [r["avg_latency"] for r in results if r.get("avg_latency") is not None]
        ttfts = [r["ttft_estimate"] for r in results if r.get("ttft_estimate") is not None]
        pct = calculate_percentiles(latencies) if latencies else {}
        consistency = None
        if pct.get("p50"):
            consistency = pct["p99"] / pct["p50"] if pct["p50"] > 0 else None

        return {
            "workload": self.name,
            "cells": len(results),
            "schema_adherence_rate": schema_adherence,
            "tool_call_correctness": tool_correctness,
            "best_ttft": min(ttfts) if ttfts else None,
            "latency_p99": pct.get("p99"),
            "latency_consistency": consistency,
        }
