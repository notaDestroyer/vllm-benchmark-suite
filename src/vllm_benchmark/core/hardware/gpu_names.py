"""GPU-name normalisation and fuzzy matching against curated tables.

``nvidia-smi`` reports names such as ``"NVIDIA A100-SXM4-80GB"``,
``"NVIDIA H100 80GB HBM3"``, ``"NVIDIA GeForce RTX 4090"`` or, on multi-GPU
hosts after :class:`~vllm_benchmark.core.server.SystemInfo` collapsing,
``"4x NVIDIA H100 80GB HBM3"``.  The scoring baselines, pricing table and
roofline spec table are keyed by short labels (``"A100-80GB"``, ``"H100"``,
``"RTX 4090"``, ``"H100 SXM"``).  Naive substring matching fails on the
first form and over-matches short labels, so every table lookup goes
through :func:`match_gpu_key`.

Author: amit
License: MIT
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

_SEP_RE = re.compile(r"[-_/,()]+")
_COUNT_PREFIX_RE = re.compile(r"^\d+x$")
_NOISE_TOKENS = frozenset({"nvidia", "geforce", "tesla", "generation", "edition", "workstation", "hbm3", "hbm2e"})


def normalize_gpu_name(name: Optional[str]) -> list[str]:
    """Return the significant lower-case tokens of a GPU name.

    Separators (``-``, ``_``, ``/``) become spaces, a leading ``"Nx"`` GPU
    count and vendor/marketing noise words are dropped, and the remaining
    tokens are returned in order.  ``None``/empty input yields ``[]``.

    >>> normalize_gpu_name("4x NVIDIA A100-SXM4-80GB")
    ['a100', 'sxm4', '80gb']
    """
    if not name:
        return []
    text = _SEP_RE.sub(" ", str(name).lower())
    tokens = [t for t in text.split() if t]
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0 and _COUNT_PREFIX_RE.match(tok):
            continue
        if tok in _NOISE_TOKENS:
            continue
        out.append(tok)
    return out


def _score(name_tokens: list[str], key_tokens: list[str]) -> Optional[float]:
    """Score how well ``key_tokens`` describe ``name_tokens`` (higher is better).

    Rules:

    * The key's first token (the model designator, e.g. ``a100``) must
      appear verbatim in the name; otherwise no match.  Exact-token
      comparison means ``l4`` never matches ``l40s``.
    * Score = number of key tokens present in the name, minus a small
      penalty per key token that is absent, so ``"A100-80GB"`` beats
      ``"A100-40GB"`` for an 80 GB card and ``"H100"`` ties with neither
      ``"H100 SXM"`` nor ``"H100 PCIe"`` being contradicted.
    * Key tokens that *contradict* the name (a different memory size
      such as ``40gb`` when the name says ``80gb``) disqualify the key.
    """
    if not key_tokens or not name_tokens:
        return None
    name_set = set(name_tokens)
    if key_tokens[0] not in name_set:
        return None
    name_mem = {t for t in name_tokens if t.endswith("gb")}
    matched = 0
    missing = 0
    for tok in key_tokens:
        if tok in name_set:
            matched += 1
        else:
            if tok.endswith("gb") and name_mem and tok not in name_mem:
                return None  # contradicting memory size
            missing += 1
    return matched - 0.1 * missing


def match_gpu_key(gpu_name: Optional[str], keys: Iterable[str]) -> Optional[str]:
    """Return the best-matching key from ``keys`` for ``gpu_name``.

    Returns ``None`` when nothing matches.  Ties are broken by key order,
    so list the preferred variant first in the table.
    """
    name_tokens = normalize_gpu_name(gpu_name)
    if not name_tokens:
        return None
    best_key: Optional[str] = None
    best_score: Optional[float] = None
    for key in keys:
        score = _score(name_tokens, normalize_gpu_name(key))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_key, best_score = key, score
    return best_key
