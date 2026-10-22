"""Fuzzy key-phrase matching as a deepeval metric.

The test case's expected_output holds key phrase requirements instead of a full
reference answer: one requirement per line, with alternative phrasings of the
same requirement separated by " | ". A requirement is met if any alternative
matches, e.g. "PII | personally identifiable information".

Alternatives are looked for in actual_output with progressively more expensive
matchers. Each matcher tries every alternative before moving on to the next,
and matching stops at the first hit:

1. exact:     the normalized phrase appears as whole words.
2. rapidfuzz: a run of words in the answer has a similarity ratio above
              RAPIDFUZZ_CUTOFF (catches typos and small inflections). Skipped
              for phrases shorter than RAPIDFUZZ_MIN_LENGTH characters.
3. n-gram:    BLEU or ROUGE over Porter-stemmed tokens (nltk) reaches
              NGRAM_CUTOFF (catches reordering and different word forms).

The score is the fraction of requirements met.
"""

import re
from collections import Counter
from typing import Optional

from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.util import ngrams
from rapidfuzz import fuzz, process

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

KEY_PHRASE_SEPARATOR = "\n"  # between requirements
ALTERNATIVE_SEPARATOR = "|"  # between alternative phrasings of one requirement
RAPIDFUZZ_CUTOFF = 85  # 0-100
# Shorter phrases skip rapidfuzz: one or two letters' difference is too large a
# share of a tiny word (e.g. "rag" vs "drag" scores 86).
RAPIDFUZZ_MIN_LENGTH = 5
NGRAM_CUTOFF = 0.6  # 0-1

_stemmer = PorterStemmer()
_smoothing = SmoothingFunction().method1


def parse_key_phrases(expected_output: str) -> list[list[str]]:
    """Requirements, each a list of alternative phrasings."""
    requirements = []
    for line in expected_output.split(KEY_PHRASE_SEPARATOR):
        alternatives = [a.strip() for a in line.split(ALTERNATIVE_SEPARATOR) if a.strip()]
        if alternatives:
            requirements.append(alternatives)
    return requirements


def format_key_phrases(requirements: list[list[str]]) -> str:
    return KEY_PHRASE_SEPARATOR.join(
        f" {ALTERNATIVE_SEPARATOR} ".join(alternatives) for alternatives in requirements
    )


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def exact_match(phrase: list[str], answer: list[str]) -> bool:
    return f" {' '.join(phrase)} " in f" {' '.join(answer)} "


def rapidfuzz_score(phrase: list[str], answer: list[str]) -> float:
    """Best similarity ratio between the phrase and any run of answer words of
    about the same length. Comparing whole-word runs avoids matching inside a
    longer word (e.g. "rag" inside "drag")."""
    n = len(phrase)
    windows = {
        " ".join(answer[i : i + size])
        for size in {max(1, n - 1), n, n + 1}
        for i in range(len(answer) - size + 1)
    }
    best = process.extractOne(" ".join(phrase), windows, scorer=fuzz.ratio)
    return best[1] if best else 0.0


def rouge_n_recall(phrase: list[str], answer: list[str], n: int) -> float:
    phrase_ngrams = Counter(ngrams(phrase, n))
    if not phrase_ngrams:
        return 0.0
    answer_ngrams = Counter(ngrams(answer, n))
    overlap = sum(min(count, answer_ngrams[gram]) for gram, count in phrase_ngrams.items())
    return overlap / sum(phrase_ngrams.values())


def bleu(phrase: list[str], answer: list[str]) -> float:
    """BLEU of the phrase against every same-length run of answer words, so the
    brevity penalty doesn't punish a short phrase for being shorter than the answer."""
    n = len(phrase)
    references = [answer[i : i + n] for i in range(len(answer) - n + 1)] or [answer]
    order = min(2, n)
    return sentence_bleu(
        references, phrase, weights=(1 / order,) * order, smoothing_function=_smoothing
    )


def ngram_score(phrase: list[str], answer: list[str]) -> tuple[str, float]:
    """The higher of BLEU and ROUGE (average of ROUGE-1 and ROUGE-2 recall for
    multi-word phrases), over stemmed tokens."""
    phrase = [_stemmer.stem(w) for w in phrase]
    answer = [_stemmer.stem(w) for w in answer]
    rouge = rouge_n_recall(phrase, answer, 1)
    if len(phrase) > 1:
        rouge = (rouge + rouge_n_recall(phrase, answer, 2)) / 2
    return max([("bleu", bleu(phrase, answer)), ("rouge", rouge)], key=lambda m: m[1])


def match_requirement(alternatives: list[str], answer: str) -> tuple[bool, str]:
    """Whether any alternative matches, and a description of how (or the best near miss)."""
    answer_words = words(answer)
    phrases = [(alt, words(alt)) for alt in alternatives if words(alt)]
    if not phrases or not answer_words:
        return False, "empty"

    for alt, phrase in phrases:
        if exact_match(phrase, answer_words):
            return True, f'"{alt}" exact'

    best_ratio = 0.0
    for alt, phrase in phrases:
        if len(" ".join(phrase)) < RAPIDFUZZ_MIN_LENGTH:
            continue
        ratio = rapidfuzz_score(phrase, answer_words)
        if ratio >= RAPIDFUZZ_CUTOFF:
            return True, f'"{alt}" rapidfuzz {ratio:.0f}'
        best_ratio = max(best_ratio, ratio)

    best_ngram = ("bleu", 0.0)
    for alt, phrase in phrases:
        name, score = ngram_score(phrase, answer_words)
        if score >= NGRAM_CUTOFF:
            return True, f'"{alt}" {name} {score:.2f}'
        best_ngram = max(best_ngram, (name, score), key=lambda m: m[1])

    return False, f"no match (rapidfuzz {best_ratio:.0f}, {best_ngram[0]} {best_ngram[1]:.2f})"


class FuzzyMatchMetric(BaseMetric):
    """Fraction of the key phrase requirements in expected_output met by actual_output.

    Succeeds when the verdict (score >= threshold) matches `expected_pass`;
    with `expected_pass=None` it has no pass/fail verdict.
    """

    def __init__(
        self,
        expected_pass: Optional[bool] = True,
        threshold: float = 0.6,
        name: str = "Fuzzy match",
    ):
        self.expected_pass = expected_pass
        self.threshold = threshold
        self.name = name
        self.include_reason = True
        self.evaluation_model = "rapidfuzz + nltk"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        requirements = parse_key_phrases(test_case.expected_output or "")
        if not requirements:
            raise ValueError("expected_output has no key phrases to match.")
        results = [
            (alternatives, *match_requirement(alternatives, test_case.actual_output or ""))
            for alternatives in requirements
        ]
        matched = sum(1 for _, ok, _ in results if ok)

        self.score = matched / len(requirements)
        self.evaluation_cost = 0
        self.success = self.is_successful()

        verdict = "pass" if self._passes_threshold() else "fail"
        expected = {True: "pass", False: "fail", None: "either"}[self.expected_pass]
        # Parentheses, not square brackets: DeepEval's results table would read
        # "[...]" as console markup and drop it.
        details = "; ".join(
            f"({f' {ALTERNATIVE_SEPARATOR} '.join(alternatives)}): {how}" for alternatives, _, how in results
        )
        self.reason = (
            f"{matched}/{len(requirements)} key phrase requirements met -> {verdict} "
            f"(threshold {self.threshold}), expected {expected}. {details}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def _passes_threshold(self) -> bool:
        return self.score is not None and self.score >= self.threshold

    def is_successful(self) -> Optional[bool]:
        if self.error is not None:
            self.success = False
        elif self.expected_pass is None:
            self.success = None
        else:
            self.success = self._passes_threshold() == self.expected_pass
        return self.success

    @property
    def __name__(self):
        return self.name
