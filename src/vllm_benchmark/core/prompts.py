"""Prompt generation utilities for benchmarking.

Provides multiple prompt generation strategies with varying cache behavior:
- classic: deterministic cybersecurity-themed text (high cache hits)
- deterministic: tokenizer-aware repetitive story (perfect cache hits)
- madlib: story with random word injection (moderate cache misses)
- random: fully random word sequences (minimal cache hits)

Author: amit
License: MIT
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Callable, Optional

from vllm_benchmark.config import DEFAULT_TOKENIZER

# ---------------------------------------------------------------------------
# Module-level lazy state for NLTK wordlist
# ---------------------------------------------------------------------------

_COMMON_WORDS: Optional[list[str]] = None


def _ensure_wordlist() -> list[str]:
    """Lazily download the NLTK Brown corpus and build the common-words list.

    This avoids downloading data at import time and caches the result for
    subsequent calls.

    Returns:
        List of the 2500 most common words from the Brown corpus.
    """
    global _COMMON_WORDS
    if _COMMON_WORDS is not None:
        return _COMMON_WORDS

    import nltk  # local import so NLTK is only loaded when needed

    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    word_freq = Counter(w.lower() for w in brown.words() if w.isalpha())
    _COMMON_WORDS = [word for word, _count in word_freq.most_common(2500)]
    return _COMMON_WORDS


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def make_random_text(ntoks: int) -> str:
    """Choose *ntoks* random words from the Brown corpus wordlist.

    Args:
        ntoks: Number of words to sample.

    Returns:
        Space-joined string of random words.
    """
    words = _ensure_wordlist()
    return " ".join(random.choice(words) for _ in range(ntoks))


def make_perturbed_story() -> str:
    """Generate one iteration of the *Wheels on the Bus* story with random word substitutions.

    Five sets of three random words replace the original onomatopoeia,
    producing a story that shares structure but differs in content across
    calls.

    Returns:
        A single perturbed story string.
    """
    rar = make_random_text(3)
    sss = make_random_text(3)
    ccc = make_random_text(3)
    bbb = make_random_text(3)
    www = make_random_text(3)

    return (
        f"The wheels on the bus go {rar}. {rar}. {rar}. "
        f"The wheels on the bus go {rar}. All through the town.. "
        f"The wipers on the bus go {sss}. {sss}. {sss}. "
        f"The wipers on the bus go {sss}. All through the town.. "
        f"The people on the bus go {ccc}. {ccc}. {ccc}. "
        f"The people on the bus go {ccc}. All through the town.. "
        f"The horn on the bus goes {bbb}. {bbb}. {bbb}. "
        f"The horn on the bus goes {bbb}. All through the town.. "
        f"The babies on the bus go {www}. {www}. {www}. "
        f"The babies on the bus go {www}. All through the town."
    )


# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------


def generate_prompt(target_tokens: int, model_name: str = "") -> str:
    """Generate a synthetic prompt of approximately *target_tokens* length.

    Uses cybersecurity threat intelligence text to simulate realistic
    workload patterns.

    Args:
        target_tokens: Approximate desired token count.
        model_name: Unused; present for interface compatibility.

    Returns:
        Generated prompt string.
    """
    base_text = (
        "Analyze the following cybersecurity threat intelligence data in detail. "
    )
    repeat_text = (
        "Advanced Persistent Threat (APT) groups continue to evolve their tactics, "
        "techniques, and procedures (TTPs) as documented in the MITRE ATT&CK framework. "
        "Nation-state actors leverage sophisticated malware campaigns targeting critical "
        "infrastructure including SCADA systems, industrial control systems, and OT networks. "
        "Recent ransomware operations demonstrate increased professionalization with affiliates "
        "using double extortion techniques, data exfiltration, and targeted attacks on backup "
        "systems. Dark web marketplaces facilitate the sale of exploits, credentials, and "
        "access to compromised networks. Vulnerability intelligence indicates zero-day exploits "
        "are being actively weaponized against enterprise systems. Threat actors employ "
        "living-off-the-land binaries (LOLBins), fileless malware, and memory-only payloads "
        "to evade detection. Network intrusion detection systems identify command and control "
        "(C2) infrastructure using domain generation algorithms (DGA) and fast-flux DNS "
        "techniques. Security operations centers analyze indicators of compromise (IOCs) "
        "including file hashes, IP addresses, and behavioral patterns to attribute attacks "
        "to specific threat actors. "
    )

    # Approximate token calculation (4 characters per token)
    base_tokens = len(base_text) // 4
    repeat_tokens = len(repeat_text) // 4
    repetitions = max(1, (target_tokens - base_tokens) // repeat_tokens)

    return base_text + (repeat_text * repetitions)


def generate_deterministic_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Create a purely deterministic prompt using a repetitive story.

    Uses a real tokenizer to precisely control token count.

    Args:
        target_tokens: Target token count for the prompt.
        tokenizer_model: HuggingFace model name for the tokenizer.

    Returns:
        Deterministic prompt string, or empty string if *target_tokens*
        is too small.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

    query_text = "Provide a concise summary of this story. "
    story_text = (
        "The wheels on the bus go round and round. Round and round. Round and round. "
        "The wheels on the bus go round and round. All through the town.. "
        "The wipers on the bus go swish, swish, swish. Swish, swish, swish. Swish, swish, swish. "
        "The wipers on the bus go swish, swish, swish. All through the town.. "
        "The people on the bus go chat, chat, chat. Chat, chat, chat. Chat, chat, chat. "
        "The people on the bus go chat, chat, chat. All through the town.. "
        "The horn on the bus goes beep, beep, beep. Beep, beep, beep. Beep, beep, beep. "
        "The horn on the bus goes beep, beep, beep. All through the town.. "
        "The babies on the bus go waa, waa, waa. Waa, waa, waa. Waa, waa, waa. "
        "The babies on the bus go waa, waa, waa. All through the town."
    )

    base_tokens = tokenizer.encode(query_text) + tokenizer.encode(story_text)
    if len(base_tokens) > target_tokens:
        return ""

    n_remaining_tokens = target_tokens - len(base_tokens)
    repetitions = n_remaining_tokens // len(tokenizer.encode(story_text))
    repeat_text = repetitions * (" " + story_text)

    return query_text + story_text + repeat_text


def generate_madlib_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Create a prompt in mad-lib style with random word injection.

    Injects randomness to promote a moderate amount of cache misses while
    maintaining structural repetition.

    Args:
        target_tokens: Target token count for the prompt.
        tokenizer_model: HuggingFace model name for the tokenizer.

    Returns:
        Mad-lib style prompt string, or empty string if *target_tokens*
        is too small.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

    query_text = "Provide a concise summary of this story. "
    story_text = make_perturbed_story()

    base_tokens = tokenizer.encode(query_text) + tokenizer.encode(story_text)
    if len(base_tokens) > target_tokens:
        return ""

    # Generate 1% fewer tokens than context length to account for
    # non-uniform tokenization of random words.
    buffer_tokens = int(target_tokens * 0.01)
    n_remaining_tokens = target_tokens - (len(base_tokens) + buffer_tokens)
    repetitions = n_remaining_tokens // len(tokenizer.encode(story_text))
    repeat_text = " ".join(make_perturbed_story() for _ in range(repetitions))

    return query_text + story_text + " " + repeat_text


def generate_random_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Create a prompt with almost entirely random text.

    Promotes a high number of cache misses by using fully random word
    sequences.

    Args:
        target_tokens: Target token count for the prompt.
        tokenizer_model: HuggingFace model name for the tokenizer.

    Returns:
        Random-text prompt string.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

    query_text = "Find the pattern in the following string. "
    # Generate 5% fewer tokens than the boundary in case the tokenizer is aggressive
    buffer_tokens = int(target_tokens * 0.05)
    n_remaining_tokens = target_tokens - (len(tokenizer.encode(query_text)) + buffer_tokens)
    random_text = make_random_text(n_remaining_tokens)

    return query_text + random_text


# ---------------------------------------------------------------------------
# Custom prompts from JSONL
# ---------------------------------------------------------------------------


def load_custom_prompts(filepath: str) -> list[str]:
    """Load prompts from a JSONL file.

    Each line in the file must be a JSON object with a ``"prompt"`` field.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of prompt strings.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        KeyError: If a line is missing the ``"prompt"`` field.
    """
    prompts: list[str] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "prompt" not in data:
                raise KeyError(
                    f"Line {line_no} in {filepath} is missing the 'prompt' field"
                )
            prompts.append(data["prompt"])
    return prompts


# ---------------------------------------------------------------------------
# Prompt generator dispatcher
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Callable] = {
    "classic": generate_prompt,
    "deterministic": generate_deterministic_prompt,
    "madlib": generate_madlib_prompt,
    "random": generate_random_prompt,
}


def get_prompt_generator(prompt_type: str) -> Callable:
    """Return the prompt generation function for the given type.

    Args:
        prompt_type: One of ``"classic"``, ``"deterministic"``,
            ``"madlib"``, or ``"random"``.

    Returns:
        The corresponding generator callable.

    Raises:
        ValueError: If *prompt_type* is not recognised.
    """
    if prompt_type not in _GENERATORS:
        raise ValueError(
            f"Unknown prompt type: {prompt_type!r}. "
            f"Choose from: {sorted(_GENERATORS)}"
        )
    return _GENERATORS[prompt_type]
