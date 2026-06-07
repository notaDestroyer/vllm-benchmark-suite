"""Packaging-level invariants for the v4 release.

These guard the user-facing contract: the version string, both console
script aliases, and the results-JSON ``schema_version`` written by the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vllm_benchmark

try:  # Python 3.11+ ships tomllib in the stdlib.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 has no stdlib TOML reader
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - optional on 3.10
        tomllib = None  # type: ignore[assignment]

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_pyproject() -> dict:
    if tomllib is None:  # pragma: no cover - only on 3.10 without tomli
        pytest.skip("no TOML reader available (Python 3.10 without tomli)")
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_version_is_4_0_0():
    assert vllm_benchmark.__version__ == "4.0.0"


def test_pyproject_version_matches_dunder():
    data = _load_pyproject()
    assert data["project"]["version"] == "4.0.0"


def test_both_console_scripts_resolve_to_cli_main():
    scripts = _load_pyproject()["project"]["scripts"]
    target = "vllm_benchmark.cli:main"
    assert scripts.get("vllm-bench") == target
    assert scripts.get("llm-bench") == target


def test_console_scripts_via_importlib_metadata():
    """If the package is installed, both entry points must resolve."""
    from importlib import metadata

    try:
        eps = metadata.entry_points(group="console_scripts")
    except metadata.PackageNotFoundError:  # pragma: no cover - not installed
        import pytest

        pytest.skip("vllm-benchmark-suite not installed; metadata unavailable")

    names = {ep.name: ep.value for ep in eps if ep.name in {"vllm-bench", "llm-bench"}}
    # When installed, both should be present and point at the CLI entry.
    if names:
        for value in names.values():
            assert value == "vllm_benchmark.cli:main"


def test_schema_version_is_4_0_in_metadata():
    """The results-JSON metadata schema_version literal is '4.0'."""
    cli_src = (Path(vllm_benchmark.__file__).parent / "cli.py").read_text()
    assert '"schema_version": "4.0"' in cli_src
    # It must be set in both serialization paths (generative + workload).
    assert cli_src.count('"schema_version": "4.0"') == 2
