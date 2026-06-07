"""Post-generation numeric verifier for the analyst report.

The language model is only permitted to render numbers that are already
present in the facts bundle.  This module checks a draft against the
canonical :func:`~vllm_benchmark.analysis.report.bundle.allowed_numbers`
set: every numeric token in the prose must match some allowed value (after
unit normalization) within a relative tolerance.  Unsupported numbers are
struck through and counted so the caller can decide whether to keep the
report or fall back to the deterministic template.

Everything here is pure — no I/O, no randomness — so it can be tested
exhaustively.

Author: amit
License: MIT
"""

from __future__ import annotations

import re
from typing import Optional

#: Matches a numeric token with optional thousands separators, decimal part
#: and a trailing unit.  Examples: ``1,234``, ``0.78``, ``78%``, ``1.2s``,
#: ``1200ms``, ``24 GB``, ``512 tok/s``.  The unit may be separated by a
#: single space.  A leading ``~`` (approximation) or ``$`` is tolerated and
#: not captured into the value.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                      # not mid-word / mid-number
    [~$]?                           # optional approx / currency marker
    (?P<num>
        \d{1,3}(?:,\d{3})+          # grouped thousands: 1,234,567
        (?:\.\d+)?
        |
        \d+(?:\.\d+)?               # plain integer or decimal
    )
    \s?
    (?P<unit>                       # longest alternatives first
        tok/s|tokens/s|docs/s|
        ms|GB|MB|TB|
        %|s|B|K|M|G
    )?
    (?!\d)                          # not immediately followed by a digit
    (?![A-Za-z])                    # ...nor a letter (would be a different unit/word)
    """,
    re.VERBOSE,
)

#: Units whose *value* must be normalized to the bundle's canonical units
#: before comparison.  Maps a matched unit to a multiplier applied to the
#: parsed value to express it in canonical terms.  ``%`` -> fraction,
#: ``ms`` -> seconds, ``B``/``K``/``M``/``G`` -> absolute counts.
_UNIT_NORMALIZERS: dict[str, float] = {
    "%": 0.01,            # 78%   -> 0.78
    "ms": 0.001,          # 1200ms -> 1.2 (seconds)
    "B": 1e9,             # 7B    -> 7e9 (billions of params)
    "G": 1e9,
    "M": 1e6,             # 500M  -> 5e8
    "K": 1e3,             # 32K   -> 32000
}


def _parse_value(raw_num: str) -> Optional[float]:
    """Parse the numeric part of a token, dropping thousands separators."""
    try:
        return float(raw_num.replace(",", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float, Optional[str]]]:
    """Find numeric tokens in ``text``.

    Handles thousands separators, decimals and trailing units (``%``,
    ``ms``, ``s``, ``GB``, ``tok/s``, and the magnitude suffixes
    ``B``/``K``/``M``/``G``).

    Args:
        text: The prose to scan.

    Returns:
        A list of ``(raw_str, value, unit)`` triples in order of
        appearance.  ``raw_str`` is the exact matched substring (including
        any unit), ``value`` is the parsed numeric value (pre-normalization)
        and ``unit`` is the matched unit or ``None``.
    """
    out: list[tuple[str, float, Optional[str]]] = []
    for m in _NUMBER_RE.finditer(text):
        value = _parse_value(m.group("num"))
        if value is None:
            continue
        out.append((m.group(0), value, m.group("unit")))
    return out


def _candidate_values(value: float, unit: Optional[str]) -> list[float]:
    """Return the canonical-unit candidates a token could faithfully denote.

    Always includes the raw value (a unitless figure compares directly).
    When a recognized unit is present, also includes the unit-normalized
    value so e.g. ``78%`` matches an allowed ``0.78`` and ``1200ms`` matches
    an allowed ``1.2``.
    """
    candidates = [value]
    if unit:
        mult = _UNIT_NORMALIZERS.get(unit)
        if mult is not None:
            candidates.append(value * mult)
    return candidates


def _matches_allowed(
    value: float,
    unit: Optional[str],
    allowed: set[float],
    tolerance: float,
) -> bool:
    """Return ``True`` if a token matches any allowed value within tolerance.

    The match is by relative tolerance (with an absolute floor so values
    near zero still compare sensibly).  Each candidate unit-normalization of
    the token is checked against every allowed value.
    """
    for cand in _candidate_values(value, unit):
        for a in allowed:
            diff = abs(cand - a)
            scale = max(abs(a), abs(cand), 1e-9)
            if diff <= tolerance * scale or diff <= 1e-6:
                return True
    return False


def verify_report(
    text: str,
    allowed_numbers: set[float],
    tolerance: float = 0.01,
) -> dict:
    """Verify every numeric token in ``text`` against the allowed set.

    Args:
        text: The draft report (Markdown).
        allowed_numbers: The canonical allowed values from
            :func:`~vllm_benchmark.analysis.report.bundle.allowed_numbers`.
        tolerance: Relative tolerance for a match.  Default 1%.

    Returns:
        A dict with:

        * ``checked`` — total numeric tokens examined,
        * ``unsupported`` — the list of raw token strings that matched no
          allowed value,
        * ``redacted_text`` — ``text`` with each unsupported token struck
          through (``~~token~~``) so a human reviewer can see what was
          rejected.

    A report with an empty ``unsupported`` list is fully grounded in the
    bundle's facts.
    """
    unsupported: list[str] = []

    # Walk matches left-to-right and rebuild the text, striking out the
    # unsupported spans.  Re-running finditer keeps span offsets exact.
    pieces: list[str] = []
    cursor = 0
    checked = 0
    for m in _NUMBER_RE.finditer(text):
        value = _parse_value(m.group("num"))
        if value is None:
            continue
        checked += 1
        raw = m.group(0)
        unit = m.group("unit")
        pieces.append(text[cursor:m.start()])
        if _matches_allowed(value, unit, allowed_numbers, tolerance):
            pieces.append(raw)
        else:
            unsupported.append(raw)
            pieces.append(f"~~{raw}~~")
        cursor = m.end()
    pieces.append(text[cursor:])

    return {
        "checked": checked,
        "unsupported": unsupported,
        "redacted_text": "".join(pieces),
    }
