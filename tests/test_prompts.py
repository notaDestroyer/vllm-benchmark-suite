"""Tests for prompt generation utilities."""

from vllm_benchmark.core.prompts import (
    generate_prompt,
    load_custom_prompts,
    get_prompt_generator,
)


def test_generate_prompt_returns_string():
    prompt = generate_prompt(1000)
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_generate_prompt_scales_with_tokens():
    short = generate_prompt(500)
    long = generate_prompt(5000)
    assert len(long) > len(short)


def test_get_prompt_generator_classic():
    gen = get_prompt_generator("classic")
    assert callable(gen)
    result = gen(1000, "")
    assert isinstance(result, str)


def test_get_prompt_generator_invalid():
    try:
        get_prompt_generator("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_load_custom_prompts(tmp_path):
    f = tmp_path / "prompts.jsonl"
    f.write_text('{"prompt": "Hello world"}\n{"prompt": "Test prompt"}\n')
    prompts = load_custom_prompts(str(f))
    assert len(prompts) == 2
    assert prompts[0] == "Hello world"
    assert prompts[1] == "Test prompt"


def test_load_custom_prompts_missing_field(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('{"text": "no prompt field"}\n')
    try:
        load_custom_prompts(str(f))
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
