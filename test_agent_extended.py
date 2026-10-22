"""Extension evals for agent.py: the agentic behaviours the main suite doesn't cover.

test_agent.py's three layers check tool-call form, answer coverage, and answer
truth. This file probes how the agent *behaves*:

- trajectory: searched before answering, bounded tool calls, converged inside MAX_STEPS
- keyword quality: the words it searched for are grounded in the query/context, not noise
- refusal: out-of-scope queries get the exact refusal -- but only *after* searching
- grounding: every answer sentence agrees with what the tool actually returned
- multi-turn: conversation history accumulates and is capped at MAX_USER_TURNS
- variance: the same query answered several times stays consistent
- coverage: one multi-part query spanning several knowledge-base sections

Like the main suite, every check is self-verifying: canned agents that
flail, guess, or refuse lazily are expected to be *caught* by their metric.

Deliberately not covered (structural, not an oversight):
- multi-tool selection -- agent.py has exactly one tool; the only "selection"
  available is search-before-answering, which the trajectory check asserts.
- real-LLM decision-making -- the default AGENT_BACKEND=extractive is
  deterministic; one llm-gated smoke test runs when AGENT_BACKEND=llm is set.

Run: deepeval test run test_agent_extended.py
Only the grounding tests need the judge; everything else is local and free.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Optional

import pytest
from rapidfuzz import fuzz

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric, JevEval
from deepeval.metrics.jev_eval import Choice
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

import agent
from deepeval_fuzzy_util import words
from deepeval_tool_util import tool_spans

from deepeval.tracing.context import current_trace_context

GOLDENS_DIR = Path(__file__).parent / "goldens"

REFUSAL_ANSWER = "Sorry I cannot help with that."
STEP_LIMIT_PREFIX = "Sorry, I couldn't reach an answer"
# Trajectory bounds: the extractive backend searches at most 3 times before
# answering/refusing; the llm backend can legally use up to MAX_STEPS - 1.
MIN_TOOL_CALLS = 1
MAX_TOOL_CALLS = 4

# Same protocol as test_agent.py: two-option Choice, chunked judge calls.
CONTRADICTION_OPTIONS = {"agrees": 1.0, "contradicts": 0.0}
GROUNDING_INSTRUCTION = 'Does this statement agree with expected_output: "{}"'
QUESTIONS_PER_CALL = 4
RESULT_SEPARATOR = "\n\n---\n\n"  # search_knowledge's separator between result blocks


def load_goldens(file_name: str) -> list[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(GOLDENS_DIR / file_name))
    return dataset.goldens


JUDGE_PASS = load_goldens("agent_judge_pass_goldens.json")
JUDGE_FAIL = load_goldens("agent_judge_fail_goldens.json")
EXTENDED = load_goldens("agent_extended_goldens.json")

MULTIPART = [g for g in EXTENDED if g.additional_metadata["kind"] == "multipart"]
REFUSALS = [g for g in EXTENDED if g.additional_metadata["scenario"] == "refuse_out_of_scope"]
ANSWERED_ANYWAY = [g for g in EXTENDED if g.additional_metadata["scenario"] == "fail_answered_anyway"]
# Queries the live agent must handle well: the main 8 plus the multi-part one.
LIVE_GOLDENS = JUDGE_PASS + MULTIPART


def case_id(golden: Golden) -> str:
    meta = golden.additional_metadata
    return f"q{meta['question_id']}-{meta['scenario']}"


def live_run(query: str) -> str:
    """One fresh agent run inside the test's trace (records the tool spans)."""
    return agent.answer_query(agent.make_client(), history=[], query=query)


def recorded_calls() -> list[ToolCall]:
    """The tool spans in the current test's trace as ToolCalls.

    Unlike deepeval_tool_util.record_tools_called this doesn't write back to
    the trace (no trace-level metrics here need it) and returns [] when no
    trace context exists, e.g. under plain pytest instead of deepeval test run.
    """
    calls = []
    for span in tool_spans():
        try:
            params = json.loads(span.input) if isinstance(span.input, str) else span.input
        except json.JSONDecodeError:
            params = {"raw": span.input}
        calls.append(ToolCall(name=span.name, input_parameters=params, output=span.output))
    return calls


def require_trace() -> None:
    """Trace-reading tests need deepeval test run's per-test trace."""
    if current_trace_context.get() is None:
        pytest.skip("no deepeval trace: run via `deepeval test run`, not plain pytest")


def tool_outputs() -> list[str]:
    """The tool outputs in the current trace -- what the agent actually saw."""
    return [s.output for s in tool_spans() if isinstance(s.output, str)]


def make_call(keywords: list[str], output: str) -> ToolCall:
    """A fake recorded tool call for canned-agent checks (no replay needed:
    the metrics below take the call list directly)."""
    return ToolCall(
        name="search_knowledge", input_parameters={"keywords": keywords}, output=output
    )


# --- Metrics ---------------------------------------------------------------


class _VerdictMetric(BaseMetric):
    """Shared plumbing: a metric succeeds when its threshold verdict matches
    `expected_pass` (None = report-only), like the other *_util metrics."""

    def __init__(self, name: str, expected_pass: Optional[bool], threshold: float):
        self.name = name
        self.expected_pass = expected_pass
        self.threshold = threshold
        self.include_reason = True
        self.evaluation_model = "rules"
        self.evaluation_cost = 0

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

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self):
        return self.name


class TrajectoryMetric(_VerdictMetric):
    """The run used a bounded, converged trajectory: between `min_calls` and
    `max_calls` tool calls and no MAX_STEPS fallback answer.

    `min_calls=1` doubles as the degenerate tool-selection check for a
    one-tool agent: it must *choose* search over answering ungrounded.
    """

    def __init__(
        self,
        calls: list[ToolCall],
        answer: str,
        min_calls: int = MIN_TOOL_CALLS,
        max_calls: int = MAX_TOOL_CALLS,
        expected_pass: Optional[bool] = True,
        threshold: float = 1.0,
        name: str = "Trajectory",
    ):
        super().__init__(name, expected_pass, threshold)
        self.calls, self.answer = calls, answer
        self.min_calls, self.max_calls = min_calls, max_calls

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        n = len(self.calls)
        problems = []
        if n < self.min_calls:
            problems.append(f"{n} tool call(s), expected at least {self.min_calls}")
        if n > self.max_calls:
            problems.append(f"{n} tool call(s), limit is {self.max_calls}")
        if self.answer.startswith(STEP_LIMIT_PREFIX):
            problems.append("hit the MAX_STEPS fallback instead of answering")
        self.score = 0.0 if problems else 1.0
        self.success = self.is_successful()
        expected = {True: "in bounds", False: "out of bounds", None: "either"}[self.expected_pass]
        self.reason = (
            f"{n} tool call(s) across the run, expected {expected}"
            + (": " + "; ".join(problems) if problems else "")
        )
        return self.score


class KeywordRelevanceMetric(_VerdictMetric):
    """Fraction of searched keyword words grounded in the query or the context.

    A keyword word counts as grounded if it appears inside a query word
    (covers stems like "summar" for "summarization") or anywhere in the
    context. Catches validly-formed but pointless searches ("zebra",
    "quantum") that the contract check in the structured layer can't see.
    """

    def __init__(
        self,
        calls: list[ToolCall],
        query: str,
        context: Optional[list[str]] = None,
        expected_pass: Optional[bool] = True,
        threshold: float = 0.5,
        name: str = "Keyword relevance",
    ):
        super().__init__(name, expected_pass, threshold)
        self.calls, self.query, self.context = calls, query, context or []

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        query_words = words(self.query)
        context_text = f" {' '.join(words(' '.join(self.context)))} "
        keyword_words = sorted(
            {
                w
                for call in self.calls
                for kw in (call.input_parameters or {}).get("keywords", [])
                for w in str(kw).lower().split()
            }
        )
        grounded = [
            w
            for w in keyword_words
            if any(w in q for q in query_words) or w in context_text
        ]
        self.score = len(grounded) / len(keyword_words) if keyword_words else 0.0
        self.success = self.is_successful()
        ungrounded = sorted(set(keyword_words) - set(grounded))
        expected = {True: "grounded", False: "ungrounded", None: "either"}[self.expected_pass]
        self.reason = (
            f"{len(grounded)}/{len(keyword_words)} keyword words grounded in query/context, "
            f"expected {expected}"
            + (f". Ungrounded: {ungrounded}" if ungrounded else "")
        )
        return self.score


class RefusalMetric(_VerdictMetric):
    """The answer is the exact refusal string (normalised); a looser
    'cannot help with that' fragment also counts, for llm-backend variance."""

    def __init__(
        self,
        expected_pass: Optional[bool] = True,
        threshold: float = 1.0,
        name: str = "Refusal",
    ):
        super().__init__(name, expected_pass, threshold)

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        answer = " ".join((test_case.actual_output or "").split()).lower()
        expected = " ".join(REFUSAL_ANSWER.split()).lower()
        ok = answer == expected or "cannot help with that" in answer
        self.score = 1.0 if ok else 0.0
        self.success = self.is_successful()
        wanted = {True: "refusal", False: "an answer", None: "either"}[self.expected_pass]
        self.reason = (
            f"answer {'is' if ok else 'is not'} a refusal, expected {wanted}: "
            f"{test_case.actual_output!r}"
        )
        return self.score


class StabilityMetric(_VerdictMetric):
    """Worst pairwise RapidFuzz similarity across N answers to the same query.

    Threshold 1.0 for the deterministic extractive backend (identical answers
    required), 0.85 for a real model (same answer, free phrasing).
    """

    def __init__(
        self,
        answers: list[str],
        expected_pass: Optional[bool] = True,
        threshold: Optional[float] = None,
        name: str = "Stability",
    ):
        super().__init__(
            name,
            expected_pass,
            threshold if threshold is not None else (1.0 if agent.BACKEND == "extractive" else 0.85),
        )
        self.answers = answers

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        worst = min(
            (fuzz.ratio(a, b) / 100 for a, b in itertools.combinations(self.answers, 2)),
            default=1.0,
        )
        problems = []
        if any(a.startswith(STEP_LIMIT_PREFIX) for a in self.answers):
            problems.append("a run hit the MAX_STEPS fallback")
        if any(a.strip() == REFUSAL_ANSWER for a in self.answers):
            problems.append("a run refused an in-scope question")
        self.score = 0.0 if problems else worst
        self.success = self.is_successful()
        self.reason = (
            f"worst pairwise similarity {worst:.0%} across {len(self.answers)} runs"
            + (": " + "; ".join(problems) if problems else "")
        )
        return self.score


class GroundingMetric(_VerdictMetric):
    """Every answer sentence agrees with its most relevant retrieved paragraph.

    The judge states use ACTUAL_OUTPUT + EXPECTED_OUTPUT, where expected_output
    is the single retrieved paragraph most overlapping that sentence -- not the
    whole retrieval. Measured behaviour on the local checkpoint: a blatant
    contradiction scores ~0.8 "agrees" against a 7-paragraph context state but
    ~0.34 against the one relevant paragraph, so each sentence gets its own
    focused state. Sentences are grouped by best paragraph, so one judge call
    covers all sentences sharing it (max QUESTIONS_PER_CALL per call).
    """

    def __init__(
        self,
        answer: str,
        retrieval: list[str],
        expected_pass: Optional[bool] = True,
        threshold: float = 0.5,
        name: str = "Grounded",
    ):
        super().__init__(name, expected_pass, threshold)
        self.answer, self.retrieval = answer, retrieval

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", self.answer.strip()) if s]
        if not sentences or not self.retrieval:
            self.score = 0.0
            self.reason = "no sentences or no retrieval to check against"
            self.success = self.is_successful()
            return self.score

        para_words = [(p, set(words(p))) for p in self.retrieval]
        groups: dict[int, list[str]] = {}
        for sentence in sentences:
            overlap = set(words(sentence))
            best = max(range(len(para_words)), key=lambda i: len(para_words[i][1] & overlap))
            groups.setdefault(best, []).append(sentence)

        outcomes, cost = [], 0.0
        for para_index, group in groups.items():
            for i in range(0, len(group), QUESTIONS_PER_CALL):
                jev = JevEval(
                    name="Grounding",
                    evaluation_params=[
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT,
                    ],
                    questions=[
                        Choice(GROUNDING_INSTRUCTION.format(s), options=CONTRADICTION_OPTIONS)
                        for s in group[i : i + QUESTIONS_PER_CALL]
                    ],
                )
                jev.measure(
                    LLMTestCase(
                        input=test_case.input,
                        actual_output=self.answer,
                        expected_output=self.retrieval[para_index],
                    ),
                    _show_indicator=False,
                )
                outcomes.extend(jev._system_one_outcomes)
                cost += jev.evaluation_cost or 0
                self.evaluation_model = jev.evaluation_model

        self.score = min(o.value for o in outcomes)
        self.evaluation_cost = cost
        self.success = self.is_successful()
        details = "; ".join(
            f"{' '.join(f'{k}={v:.2f}' for k, v in o.probabilities.items())}"
            for o in outcomes
        )
        self.reason = (
            f"min of {len(outcomes)} sentence(s) each checked against its "
            f"best-matching retrieved paragraph -> {self.score:.2f} "
            f"(threshold {self.threshold}). Probabilities: {details}"
        )
        return self.score


def retrieval_blocks() -> list[str]:
    """The individual paragraphs the tool returned in the current trace."""
    return [
        block
        for output in tool_outputs()
        for block in output.split(RESULT_SEPARATOR)[1:]
    ]


# --- Trajectory + keyword quality -------------------------------------------


@pytest.mark.parametrize("golden", LIVE_GOLDENS, ids=case_id)
def test_trajectory_and_keywords(golden: Golden):
    """Live run: the agent searches before answering, stays within the call
    budget, converges, and its keywords are grounded in the query/context."""
    require_trace()
    answer = live_run(golden.input)
    calls = recorded_calls()
    assert_test(
        LLMTestCase(
            input=golden.input,
            actual_output=answer,
            expected_output=golden.expected_output,
        ),
        [
            TrajectoryMetric(calls, answer),
            KeywordRelevanceMetric(calls, golden.input, golden.context),
        ],
    )


def test_trajectory_flail_caught():
    """A canned agent spamming the tool past the budget must fail the check."""
    calls = [make_call(["pii"], "No paragraphs matched: pii") for _ in range(MAX_TOOL_CALLS + 1)]
    assert_test(
        LLMTestCase(input=JUDGE_PASS[0].input, actual_output="(flailed)"),
        [TrajectoryMetric(calls, "(flailed)", expected_pass=False)],
    )


def test_lazy_refusal_caught():
    """Refusing without ever checking the knowledge base must fail: the right
    words alone aren't enough -- the agent has to look first."""
    assert_test(
        LLMTestCase(input=REFUSALS[0].input, actual_output=REFUSAL_ANSWER),
        [
            RefusalMetric(expected_pass=True),
            TrajectoryMetric([], REFUSAL_ANSWER, expected_pass=False, name="Searched first"),
        ],
    )


def test_keyword_relevance_irrelevant_caught():
    """Well-formed keywords unrelated to the query must fail relevance."""
    golden = JUDGE_PASS[0]
    calls = [make_call(["zebra", "quantum"], "No paragraphs matched: zebra, quantum")]
    assert_test(
        LLMTestCase(input=golden.input, actual_output="(mocked)"),
        [KeywordRelevanceMetric(calls, golden.input, golden.context, expected_pass=False)],
    )


# --- Refusal path ------------------------------------------------------------


@pytest.mark.parametrize("golden", REFUSALS, ids=case_id)
def test_out_of_scope_refusal(golden: Golden):
    """Out-of-scope query: exact refusal, and only after searching the KB."""
    require_trace()
    answer = live_run(golden.input)
    calls = recorded_calls()
    assert_test(
        LLMTestCase(
            input=golden.input,
            actual_output=answer,
            expected_output=golden.expected_output,
        ),
        [
            RefusalMetric(expected_pass=True),
            TrajectoryMetric(calls, answer, max_calls=3, name="Searched first"),
        ],
    )


@pytest.mark.parametrize("golden", ANSWERED_ANYWAY, ids=case_id)
def test_out_of_scope_answered_anyway_caught(golden: Golden):
    """A canned agent that answers an out-of-scope question anyway must be
    caught by the refusal check."""
    assert_test(
        LLMTestCase(
            input=golden.input,
            actual_output=golden.actual_output,
            expected_output=golden.expected_output,
        ),
        [RefusalMetric(expected_pass=False)],
    )


# --- Grounding: answer vs. what the tool returned ----------------------------


@pytest.mark.parametrize("golden", LIVE_GOLDENS, ids=case_id)
def test_grounding(golden: Golden):
    """Every sentence of the live answer agrees with the tool outputs the
    agent actually retrieved (faithfulness). Each sentence is checked against
    its best-matching retrieved paragraph."""
    require_trace()
    answer = live_run(golden.input)
    retrieval = retrieval_blocks() or (golden.context or [])
    assert_test(
        LLMTestCase(input=golden.input, actual_output=answer),
        [GroundingMetric(answer, retrieval, expected_pass=True)],
    )


@pytest.mark.parametrize("golden", JUDGE_FAIL, ids=case_id)
def test_grounding_catches_unsupported(golden: Golden):
    """Canned answers checked against the knowledge they 'retrieved':
    contradictions must fail grounding; thin-but-true answers should still
    count as grounded (grounding != completeness)."""
    expected = {"fail_contradiction": False, "fail_incomplete": True}[
        golden.additional_metadata["scenario"]
    ]
    assert_test(
        LLMTestCase(input=golden.input, actual_output=golden.actual_output),
        [GroundingMetric(golden.actual_output, golden.context or [], expected_pass=expected)],
    )


# --- Variance ---------------------------------------------------------------


@pytest.mark.parametrize("golden", LIVE_GOLDENS, ids=case_id)
def test_answer_stability(golden: Golden):
    """Three fresh runs of the same query produce consistent answers
    (identical under extractive, >=85% similar under a real model)."""
    answers = [live_run(golden.input) for _ in range(3)]
    assert_test(
        LLMTestCase(input=golden.input, actual_output=answers[0]),
        [StabilityMetric(answers)],
    )


# --- Multi-turn memory (plain pytest asserts -- no metrics needed) -----------


def test_multi_turn_history_accumulates():
    """Two queries on one history: both turns are kept, oldest first."""
    client, history = agent.make_client(), []
    agent.answer_query(client, history, JUDGE_PASS[0].input)
    agent.answer_query(client, history, JUDGE_PASS[1].input)
    assert [turn[0]["content"] for turn in history] == [
        JUDGE_PASS[0].input,
        JUDGE_PASS[1].input,
    ]


def test_multi_turn_window_capped():
    """After MAX_USER_TURNS+2 queries the history still holds only the last
    MAX_USER_TURNS turns -- the context window is enforced."""
    client, history = agent.make_client(), []
    for _ in range(agent.MAX_USER_TURNS + 2):
        agent.answer_query(client, history, JUDGE_PASS[0].input)
    assert len(history) == agent.MAX_USER_TURNS


@pytest.mark.skipif(
    agent.BACKEND != "llm",
    reason="needs AGENT_BACKEND=llm; the extractive stand-in can't resolve 'it'",
)
def test_multi_turn_followup():
    """A pronoun follow-up resolves against earlier turns with a real model."""
    client, history = agent.make_client(), []
    agent.answer_query(client, history, "What does PII include?")
    answer = agent.answer_query(client, history, "Give me two examples of it.")
    assert REFUSAL_ANSWER.lower() not in answer.lower()


# --- Real-LLM smoke test ------------------------------------------------------


@pytest.mark.skipif(
    agent.BACKEND != "llm",
    reason="only meaningful with AGENT_BACKEND=llm and a configured INFERENCE_* endpoint",
)
def test_llm_backend_smoke():
    """With a real model configured, the live agent still searches and
    answers -- the cheapest end-to-end check of actual LLM tool choice."""
    answer = live_run(JUDGE_PASS[0].input)
    calls = recorded_calls()
    assert calls, "the llm agent answered without calling search_knowledge"
    assert answer and not answer.startswith(STEP_LIMIT_PREFIX)
