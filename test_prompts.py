from vllm_benchmark_suitev2 import generate_prompt, generate_deterministic_prompt, generate_madlib_prompt, generate_random_prompt
from transformers import AutoTokenizer

TOKENIZER = "google/gemma-3-4b-it"

def test_existing_prompt():
    """Tests the default prompt function to ensure it always produces prompts under the token limit."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for N in [1000, 2000, 4000, 8000, 160000, 320000, 64000, 1280000]:
        prompt = generate_prompt(N, TOKENIZER)
        ntokens = len(tokenizer.encode(prompt)) #use Qwen's tokenizer -- default tokenizer for benchmark
        print(f"For N={N}, generated={ntokens}. Prompt looks like: '{prompt[:100]}'...")
        assert ntokens <= N

def test_deterministic_prompt():
    """Tests our high-cache hit prompt function to ensure it always produces prompts under the token limit."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for N in [1000, 2000, 4000, 8000, 160000, 320000, 64000, 1280000]:
        prompt = generate_deterministic_prompt(N, TOKENIZER)
        ntokens = len(tokenizer.encode(prompt)) #use Qwen's tokenizer -- default tokenizer for benchmark
        print(f"For N={N}, generated={ntokens}. Prompt looks like: '{prompt[:100]}'...")
        assert ntokens <= N

def test_madlib_prompt():
    """Tests our high-cache hit prompt function to ensure it always produces prompts under the token limit."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for N in [1000, 2000, 4000, 8000, 160000, 320000, 64000, 1280000]:
        prompt = generate_madlib_prompt(N, TOKENIZER)
        ntokens = len(tokenizer.encode(prompt)) #use Qwen's tokenizer -- default tokenizer for benchmark
        print(f"For N={N}, generated={ntokens}. Prompt looks like: '{prompt[:100]}'...")
        assert ntokens <= N

def test_random_prompt():
    """Tests our low-cache hit prompt function to ensure it always produces prompts under the token limit."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for N in [1000, 2000, 4000, 8000, 160000, 320000, 64000, 1280000]:
        prompt = generate_random_prompt(N, TOKENIZER)
        ntokens = len(tokenizer.encode(prompt)) #use Qwen's tokenizer -- default tokenizer for benchmark
        print(f"For N={N}, generated={ntokens}. Prompt looks like: '{prompt[:100]}'...")
        assert ntokens <= N

def test_similar_lens():
    """Tests that all prompt generation functions make similar length prompts."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        token_tolerance = int(N * 0.5) + 200 #should be within 5% -- this is the random generator's lower bound.
        ctokens = len(tokenizer.encode(generate_prompt(N, TOKENIZER))) #generate prompt, tokenize, and get length
        dtokens = len(tokenizer.encode(generate_deterministic_prompt(N, TOKENIZER)))
        mltokens = len(tokenizer.encode(generate_madlib_prompt(N, TOKENIZER)))
        rtokens = len(tokenizer.encode(generate_random_prompt(N, TOKENIZER)))
        print(f"(N={N}) ctokens={ctokens},dtokens={dtokens},mltokens={mltokens},rtokens={rtokens}") #NOTE: we don't explicitly test the classic token count, this is for demonstration
        print(f"(N={N}) tolerance={token_tolerance}, max difference={max(dtokens, mltokens, rtokens)-min(dtokens, mltokens, rtokens)}")
        assert max(dtokens, mltokens, rtokens) - min(dtokens, mltokens, rtokens) < token_tolerance

if __name__ == "__main__":
    pass

    print(f"TESTING THE EXISTING PROMPT")
    test_existing_prompt()
    
    print(f"TESTING DETERMINISTIC PROMPT")
    test_deterministic_prompt()

    print(f"TESTING MADLIB PROMPT")
    test_madlib_prompt()

    print(f"TESTING RANDOM PROMPT")
    test_random_prompt()

    print(f"TESTING RELATIVE PROMPT SIZES")
    test_similar_lens()