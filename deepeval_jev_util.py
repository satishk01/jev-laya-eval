import asyncio
from typing import Optional

from deepeval.metrics import BaseMetric, JevEval
from deepeval.test_case import LLMTestCase


class SharedJevDecision:
    """Runs JevEval once per test case and shares its per-question outcomes."""

    def __init__(self, jev: JevEval):
        self.jev = jev
        self._decided_key = None
        self._lock: Optional[asyncio.Lock] = None

    @staticmethod
    def _key(test_case: LLMTestCase):
        return (test_case.input, test_case.actual_output, test_case.expected_output)

    def decide(self, test_case: LLMTestCase) -> bool:
        """Returns True if this call made the Jev request."""
        if self._decided_key == self._key(test_case):
            return False
        self.jev.measure(test_case, _show_indicator=False)
        self._decided_key = self._key(test_case)
        return True

    async def a_decide(self, test_case: LLMTestCase) -> bool:
        # deepeval measures metrics concurrently, so the lock makes the
        # second metric wait for the first one's Jev call instead of repeating it.
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._decided_key == self._key(test_case):
                return False
            await self.jev.a_measure(test_case, _show_indicator=False)
            self._decided_key = self._key(test_case)
            return True


class JevQuestionMetric(BaseMetric):
    """One or more questions' results from shared Jev calls, reported as a metric.

    `decisions` is a SharedJevDecision or a list of them (a list when the
    questions were split across calls to stay under the server's per-request
    limit); `question_index` is an int or a list of ints into the
    concatenated outcomes, whose values are combined with min(). Succeeds
    when the verdict (score >= threshold) matches `expected_pass`; with
    `expected_pass=None` it has no pass/fail verdict.
    """

    def __init__(
        self,
        decisions: "SharedJevDecision | list[SharedJevDecision]",
        question_index: "int | list[int]",
        name: str,
        expected_pass: Optional[bool],
        threshold: float = 0.5,
    ):
        self.decisions = decisions if isinstance(decisions, list) else [decisions]
        self.question_index = (
            question_index if isinstance(question_index, list) else [question_index]
        )
        self.name = name
        self.expected_pass = expected_pass
        self.threshold = threshold
        self.include_reason = True
        self.evaluation_model = self.decisions[0].jev.evaluation_model

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        made = [d.decide(test_case) for d in self.decisions]
        self._finalize(made_call=any(made))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        # Sequential: the lock serialises them anyway, and overlapping
        # requests would just queue on the judge.
        made = [await d.a_decide(test_case) for d in self.decisions]
        self._finalize(made_call=any(made))
        return self.score

    def _finalize(self, made_call: bool) -> None:
        # WARNING `_system_one_outcomes` is not a public API,
        # so this implementation is brittle
        outcomes = [o for d in self.decisions for o in d.jev._system_one_outcomes]
        picked = [outcomes[i] for i in self.question_index]
        self.score = min(o.value for o in picked)
        # Only a metric that made the Jev call carries its cost.
        self.evaluation_cost = (
            sum(d.jev.evaluation_cost or 0 for d in self.decisions) if made_call else 0
        )
        self.success = self.is_successful()

        verdict = "pass" if self._passes_threshold() else "fail"
        expected = {True: "pass", False: "fail", None: "either"}[self.expected_pass]
        details = "; ".join(
            f"{o.question} ({', '.join(f'{k}={v:.2f}' for k, v in o.probabilities.items())})"
            for o in picked
        )
        self.reason = (
            f"min of {len(picked)} question(s) -> {verdict} (threshold {self.threshold}), "
            f"expected {expected}. Probabilities: {details}"
        )

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
