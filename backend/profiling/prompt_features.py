"""
Prompt Feature Extraction for Energy Prediction.

Extracts textual/linguistic features from prompts that correlate with energy
consumption during inference. Based on findings from Caravaca et al. 2025
"From Prompts to Power" — prompt complexity metrics significantly improve
energy prediction accuracy (R² improvement of 0.05-0.12).

Key insight: Prompts with higher lexical diversity, longer sentences, and
more complex structure tend to require more energy during both prefill
(due to attention pattern complexity) and decode (due to longer generations).

EP-101: Prompt Feature Extraction
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class PromptFeatures:
    """Extracted features from a prompt text."""

    # Length features
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float  # words per sentence

    # Lexical diversity
    type_token_ratio: float  # unique words / total words
    hapax_ratio: float  # words appearing exactly once / total words

    # Structural features
    question_count: int  # number of question marks
    instruction_density: float  # imperative/instruction keywords per sentence
    code_block_count: int  # fenced code blocks (```)
    list_item_count: int  # bullet/numbered list items
    has_system_prompt: bool  # whether it looks like a system prompt

    # Complexity features
    avg_token_entropy: float  # character-level entropy (proxy for surprisal)
    punctuation_density: float  # punctuation chars / total chars
    uppercase_ratio: float  # uppercase chars / alpha chars
    whitespace_ratio: float  # whitespace / total chars
    special_char_ratio: float  # non-alphanumeric, non-space / total chars

    # Vocabulary features
    long_word_ratio: float  # words > 8 chars / total words
    numeric_token_ratio: float  # numeric tokens / total tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_feature_vector(self) -> List[float]:
        """
        Convert to ordered feature vector for ML model input.
        Order must match feature_names().
        """
        return [
            self.char_count,
            self.word_count,
            self.sentence_count,
            self.avg_word_length,
            self.avg_sentence_length,
            self.type_token_ratio,
            self.hapax_ratio,
            self.question_count,
            self.instruction_density,
            self.code_block_count,
            self.list_item_count,
            1.0 if self.has_system_prompt else 0.0,
            self.avg_token_entropy,
            self.punctuation_density,
            self.uppercase_ratio,
            self.whitespace_ratio,
            self.special_char_ratio,
            self.long_word_ratio,
            self.numeric_token_ratio,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Feature names in the same order as to_feature_vector()."""
        return [
            "prompt_char_count",
            "prompt_word_count",
            "prompt_sentence_count",
            "prompt_avg_word_length",
            "prompt_avg_sentence_length",
            "prompt_type_token_ratio",
            "prompt_hapax_ratio",
            "prompt_question_count",
            "prompt_instruction_density",
            "prompt_code_block_count",
            "prompt_list_item_count",
            "prompt_has_system_prompt",
            "prompt_avg_token_entropy",
            "prompt_punctuation_density",
            "prompt_uppercase_ratio",
            "prompt_whitespace_ratio",
            "prompt_special_char_ratio",
            "prompt_long_word_ratio",
            "prompt_numeric_token_ratio",
        ]


# Imperative / instruction keywords (lowercase)
_INSTRUCTION_KEYWORDS = frozenset([
    "write", "create", "generate", "explain", "describe", "list",
    "summarize", "translate", "analyze", "compare", "implement",
    "build", "design", "calculate", "solve", "find", "show",
    "tell", "give", "provide", "make", "help", "define",
    "convert", "extract", "output", "return", "print", "format",
    "rewrite", "refactor", "optimize", "debug", "fix", "review",
])

# System prompt indicators
_SYSTEM_INDICATORS = frozenset([
    "you are", "you're a", "your role", "act as", "behave as",
    "system:", "instructions:", "your task", "as an ai",
    "you will", "you should", "you must",
])


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _char_entropy(text: str) -> float:
    """Calculate character-level Shannon entropy."""
    if not text:
        return 0.0
    freq = Counter(text.lower())
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_prompt_features(text: str) -> PromptFeatures:
    """
    Extract linguistic and structural features from a prompt.

    Args:
        text: The raw prompt text.

    Returns:
        PromptFeatures dataclass with all extracted features.
    """
    if not text or not text.strip():
        return PromptFeatures(
            char_count=0, word_count=0, sentence_count=0,
            avg_word_length=0.0, avg_sentence_length=0.0,
            type_token_ratio=0.0, hapax_ratio=0.0,
            question_count=0, instruction_density=0.0,
            code_block_count=0, list_item_count=0,
            has_system_prompt=False, avg_token_entropy=0.0,
            punctuation_density=0.0, uppercase_ratio=0.0,
            whitespace_ratio=0.0, special_char_ratio=0.0,
            long_word_ratio=0.0, numeric_token_ratio=0.0,
        )

    # Basic counts
    char_count = len(text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words) if words else 1
    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)

    # Word-level features
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = sum(word_lengths) / len(word_lengths)
    avg_sentence_length = word_count / sentence_count

    # Lexical diversity
    words_lower = [w.lower() for w in words]
    unique_words = set(words_lower)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

    word_freq = Counter(words_lower)
    hapax_count = sum(1 for c in word_freq.values() if c == 1)
    hapax_ratio = hapax_count / word_count if word_count > 0 else 0.0

    # Structural features
    question_count = text.count('?')

    instruction_words = sum(
        1 for w in words_lower if w in _INSTRUCTION_KEYWORDS
    )
    instruction_density = instruction_words / sentence_count

    code_block_count = text.count('```') // 2  # pairs of fences
    list_item_count = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
    list_item_count += len(re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE))

    text_lower = text.lower()
    has_system_prompt = any(ind in text_lower for ind in _SYSTEM_INDICATORS)

    # Complexity features
    avg_token_entropy = _char_entropy(text)

    punct_chars = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\@#$%^&*~`')
    punctuation_density = punct_chars / char_count if char_count > 0 else 0.0

    alpha_chars = sum(1 for c in text if c.isalpha())
    upper_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = upper_chars / alpha_chars if alpha_chars > 0 else 0.0

    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_chars / char_count if char_count > 0 else 0.0

    special_chars = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    )
    special_char_ratio = special_chars / char_count if char_count > 0 else 0.0

    # Vocabulary features
    long_words = sum(1 for w in words if len(w) > 8)
    long_word_ratio = long_words / word_count if word_count > 0 else 0.0

    numeric_tokens = sum(1 for w in words if re.match(r'^\d+\.?\d*$', w))
    numeric_token_ratio = numeric_tokens / word_count if word_count > 0 else 0.0

    return PromptFeatures(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=avg_word_length,
        avg_sentence_length=avg_sentence_length,
        type_token_ratio=type_token_ratio,
        hapax_ratio=hapax_ratio,
        question_count=question_count,
        instruction_density=instruction_density,
        code_block_count=code_block_count,
        list_item_count=list_item_count,
        has_system_prompt=has_system_prompt,
        avg_token_entropy=avg_token_entropy,
        punctuation_density=punctuation_density,
        uppercase_ratio=uppercase_ratio,
        whitespace_ratio=whitespace_ratio,
        special_char_ratio=special_char_ratio,
        long_word_ratio=long_word_ratio,
        numeric_token_ratio=numeric_token_ratio,
    )


def compare_prompt_features(
    prompts: List[str],
) -> List[Dict[str, Any]]:
    """
    Extract and compare features across multiple prompts.
    Useful for understanding which prompts are likely to be energy-intensive.

    Args:
        prompts: List of prompt texts.

    Returns:
        List of feature dictionaries, one per prompt.
    """
    results = []
    for i, prompt in enumerate(prompts):
        features = extract_prompt_features(prompt)
        d = features.to_dict()
        d["prompt_index"] = i
        d["prompt_preview"] = prompt[:100] + ("..." if len(prompt) > 100 else "")
        results.append(d)
    return results
