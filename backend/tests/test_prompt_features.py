"""
Tests for prompt feature extraction (EP-101).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from profiling.prompt_features import (
    extract_prompt_features,
    compare_prompt_features,
    PromptFeatures,
)


def test_empty_prompt():
    """Empty/whitespace prompts should return zero features."""
    f = extract_prompt_features("")
    assert f.char_count == 0
    assert f.word_count == 0
    assert f.type_token_ratio == 0.0

    f2 = extract_prompt_features("   ")
    assert f2.char_count == 0


def test_simple_prompt():
    """Basic prompt should extract reasonable features."""
    f = extract_prompt_features("What is the capital of France?")
    assert f.char_count > 0
    assert f.word_count == 6
    assert f.sentence_count == 1
    assert f.question_count == 1
    assert f.avg_word_length > 0
    assert 0 < f.type_token_ratio <= 1.0


def test_complex_prompt():
    """Complex prompt with code blocks, lists, and instructions."""
    prompt = """You are a Python expert. Write a function that:
- Takes a list of integers
- Returns the sum of even numbers
- Handles empty lists gracefully

```python
def sum_even(nums):
    pass
```

Explain your solution step by step. What is the time complexity?"""

    f = extract_prompt_features(prompt)
    assert f.code_block_count == 1
    assert f.list_item_count >= 3
    assert f.has_system_prompt is True  # "You are"
    assert f.question_count >= 1
    assert f.instruction_density > 0  # "write", "explain"


def test_instruction_detection():
    """Instruction keywords should be detected."""
    f = extract_prompt_features("Write a summary. Explain the concept. Compare both approaches.")
    assert f.instruction_density > 0

    f2 = extract_prompt_features("The weather is nice today.")
    assert f2.instruction_density == 0


def test_system_prompt_detection():
    """System prompts should be detected."""
    assert extract_prompt_features("You are a helpful assistant.").has_system_prompt is True
    assert extract_prompt_features("Act as a doctor.").has_system_prompt is True
    assert extract_prompt_features("Tell me a joke.").has_system_prompt is False


def test_lexical_diversity():
    """Repeated text should have lower TTR than diverse text."""
    repeated = "the the the the the the the the"
    diverse = "complex algorithm optimization parallel distributed computing throughput latency"

    f_rep = extract_prompt_features(repeated)
    f_div = extract_prompt_features(diverse)

    assert f_rep.type_token_ratio < f_div.type_token_ratio


def test_entropy():
    """Entropy should be higher for more diverse text."""
    simple = "aaaaaa"
    complex_text = "aZ3!xK9@qW"

    f_simple = extract_prompt_features(simple)
    f_complex = extract_prompt_features(complex_text)

    assert f_complex.avg_token_entropy > f_simple.avg_token_entropy


def test_feature_vector_length():
    """Feature vector should match feature names length."""
    f = extract_prompt_features("Write a test function.")
    vec = f.to_feature_vector()
    names = PromptFeatures.feature_names()

    assert len(vec) == len(names)
    assert len(vec) == 19  # Known count


def test_to_dict():
    """to_dict should return all fields."""
    f = extract_prompt_features("Hello world!")
    d = f.to_dict()

    assert "char_count" in d
    assert "type_token_ratio" in d
    assert "avg_token_entropy" in d
    assert isinstance(d, dict)


def test_compare_prompts():
    """compare_prompt_features should return list of dicts."""
    prompts = [
        "What is 2+2?",
        "Write a comprehensive essay about the history of artificial intelligence, covering the key developments from the 1950s to present day.",
    ]
    results = compare_prompt_features(prompts)

    assert len(results) == 2
    assert results[0]["prompt_index"] == 0
    assert results[1]["prompt_index"] == 1
    # Longer prompt should have higher char count
    assert results[1]["char_count"] > results[0]["char_count"]
    assert "prompt_preview" in results[0]


def test_numeric_tokens():
    """Numeric token detection."""
    f = extract_prompt_features("Calculate 42 plus 17 and multiply by 3.14")
    assert f.numeric_token_ratio > 0


def test_long_words():
    """Long word ratio detection."""
    f = extract_prompt_features("internationalization standardization")
    assert f.long_word_ratio > 0

    f2 = extract_prompt_features("a to be or not")
    assert f2.long_word_ratio == 0


if __name__ == "__main__":
    tests = [
        test_empty_prompt,
        test_simple_prompt,
        test_complex_prompt,
        test_instruction_detection,
        test_system_prompt_detection,
        test_lexical_diversity,
        test_entropy,
        test_feature_vector_length,
        test_to_dict,
        test_compare_prompts,
        test_numeric_tokens,
        test_long_words,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        exit(1)
