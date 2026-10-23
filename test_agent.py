"""Evals for agent.py, grouped by query and run as a sequence of layers.

Run with `uv run deepeval test run test_agent.py`: the structured layer needs
the per-test trace that DeepEval's pytest plugin sets up.

Each query has goldens in every enabled layer, run in this order, cheapest first:
1. structured: rule-based validation of the agent's tool calls
2. fuzzy:      key-phrase matching of the answer (exact, RapidFuzz, BLEU/ROUGE)
3. judge:      Jev checks the answer against a reference answer, one API call per test

Within a layer, the live agent is expected to pass, and mocked agents that
simulate each failure mode are expected to be caught. With SEQUENCE_FAIL_FAST,
once any test in a layer fails for a query, that query's later layers are skipped.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import JevEval, ToolCorrectnessMetric
from deepeval.metrics.jev_eval import Choice, Score
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.tracing import observe, update_current_span
from deepeval.tracing.context import current_trace_context

import agent
from deepeval_fuzzy_util import FuzzyMatchMetric
from deepeval_jev_util import JevQuestionMetric, SharedJevDecision
from deepeval_tool_util import ExpectedVerdictMetric, attach_tool_metrics, record_tools_called

# Toggle each layer of evals.
RUN_STRUCTURED_EVALS = True  # validation of the agent's tool calls, via @observe spans
RUN_FUZZY_EVALS = True  # key-phrase matching of the answer: exact, RapidFuzz, BLEU/ROUGE
RUN_JEV_EVALS = True  # the judge: Jev compares the answer with a reference answer
# Once a layer fails for a query, skip that query's later layers.
SEQUENCE_FAIL_FAST = True

GOLDENS_DIR = Path(__file__).parent / "goldens"


def load_goldens(file_name: str) -> list[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(GOLDENS_DIR / file_name))
    return dataset.goldens


# Whether each metric should pass for each scenario; None means not checked.
STRUCTURED_EXPECTED_OUTCOMES = {
    "pass": {"Tool call validation": True, "Tool correctness": True},
    "fail_invalid_keywords": {"Tool call validation": False, "Tool correctness": True},
    "fail_irrelevant_results": {"Tool call validation": False, "Tool correctness": True},
    "fail_no_tool_call": {"Tool call validation": None, "Tool correctness": False},
}

# The judge questions are built per test case in assert_jev_correctness:
# one contradiction Choice per answer sentence (see below), plus a
# completeness Score. Contradiction is checked per sentence because a
# whole-answer judgement lets a fluent answer with one blatant false
# claim pass -- laya scores "basically the same setting" ~0.7 agrees.
# Checking each sentence against expected_output and taking the worst is
# the standard decompose-and-verify approach: one bad sentence sinks it.
#
# Each sentence is asked as a two-option Choice, not a Noul: laya's noul
# head picks the false/true label token instead of reading the state
# (laya#156), and a Choice's value over these credits is the same
# P(statement holds).
CONTRADICTION_INSTRUCTION = 'Does this statement agree with expected_output: "{}"'
CONTRADICTION_OPTIONS = {"agrees": 1.0, "contradicts": 0.0}
COMPLETENESS_QUESTION = Score(
    "How much of the detail in expected_output is present in actual_output?",
    levels=["Almost none", "Some", "Most", "All of it"],
)
# The local judge encodes the states once per question in a call, so calls
# are capped at a few questions to stay well under the request timeout.
QUESTIONS_PER_CALL = 4

# Whether each judge metric should pass for each scenario. A metric missing
# from a scenario is reported with its score, but doesn't decide pass/fail.
EXPECTED_OUTCOMES = {
    "pass": {"No contradiction": True, "Completeness": True},
    "fail_contradiction": {"No contradiction": False},
    # fail_incomplete's Completeness is measured but doesn't decide: laya's
    # score head reads topical overlap, not coverage, so it rates a thin
    # on-topic answer ~0.7 (hosted jev-1.13 did catch these). The fuzzy
    # layer's key-phrase check is what gates incomplete answers now.
    "fail_incomplete": {"No contradiction": True, "Completeness": None},
}

_answers: dict[str, str] = {}


def run_agent(query: str, fresh: bool = False) -> str:
    """agent.py's answer to a single query, with no earlier conversation.

    Answers are cached, so later layers evaluate the same run as earlier ones. `fresh=True` always runs the agent, which the
    structured layer needs: a cached answer records no spans in the current
    test's trace.
    """
    if fresh or query not in _answers:
        _answers[query] = agent.answer_query(agent.make_client(), history=[], query=query)
    return _answers[query]


def mocked_agent(golden: Golden):
    """An agent that simulates the golden's failure mode by giving its incorrect answer."""

    def answer(query: str) -> str:
        assert query == golden.input
        return golden.actual_output

    return answer


@observe(type="tool", name="search_knowledge")
def replay_tool_call(keywords: list[str], output: str) -> str:
    update_current_span(input=agent.to_json({"keywords": keywords}), output=output)
    return output


@observe(type="agent", name="mocked_agent")
def replay_tool_calls(golden: Golden) -> str:
    """A mocked agent that simulates the golden's failure mode by making its
    recorded tool calls, traced as tool spans just like agent.py's."""
    update_current_span(input=golden.input)
    for call in golden.tools_called or []:
        replay_tool_call(call.input_parameters["keywords"], call.output)
    return "(mocked answer)"


def assert_tool_calls(golden: Golden) -> None:
    # Each tool span in the test's trace gets a validation metric, and the
    # trace gets a tool correctness metric comparing tools_called with
    # the golden's expected_tools.
    expected = STRUCTURED_EXPECTED_OUTCOMES[golden.additional_metadata["scenario"]]
    attach_tool_metrics(
        max_results=agent.MAX_SEARCH_RESULTS, expected_pass=expected["Tool call validation"]
    )
    record_tools_called()
    tool_correctness = ExpectedVerdictMetric(
        ToolCorrectnessMetric(threshold=1.0),
        expected_pass=expected["Tool correctness"],
        name="Tool correctness",
    )
    assert_test(golden=golden, metrics=[tool_correctness])


def make_test_case(golden: Golden, actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        context=golden.context,
    )


def assert_fuzzy_match(golden: Golden, actual_output: str) -> None:
    expected_pass = golden.additional_metadata["expected_verdict"] == "pass"
    assert_test(make_test_case(golden, actual_output), [FuzzyMatchMetric(expected_pass=expected_pass)])


def assert_jev_correctness(golden: Golden, actual_output: str) -> None:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", actual_output.strip()) if s]
    questions = [
        Choice(CONTRADICTION_INSTRUCTION.format(s), options=CONTRADICTION_OPTIONS)
        for s in sentences
    ]
    questions.append(COMPLETENESS_QUESTION)

    # Questions go to Jev in chunks: every question in a call costs the model
    # a pass over the states, and a single call with all of them can exceed
    # the request timeout on a CPU-resident checkpoint.
    decisions = [
        SharedJevDecision(
            JevEval(
                name="Correctness",
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                questions=questions[i : i + QUESTIONS_PER_CALL],
            )
        )
        for i in range(0, len(questions), QUESTIONS_PER_CALL)
    ]

    expected = EXPECTED_OUTCOMES[golden.additional_metadata["scenario"]]
    metrics = [
        JevQuestionMetric(
            decisions,
            list(range(len(sentences))),  # min over every sentence's verdict
            "No contradiction",
            expected_pass=expected.get("No contradiction"),
        ),
        JevQuestionMetric(
            decisions,
            len(questions) - 1,
            "Completeness",
            expected_pass=expected.get("Completeness"),
        ),
    ]
    assert_test(make_test_case(golden, actual_output), metrics)


def answer_for(golden: Golden) -> str:
    """The live agent's answer for a pass golden, a mocked agent's otherwise."""
    if golden.additional_metadata["scenario"] == "pass":
        return run_agent(golden.input)
    return mocked_agent(golden)(golden.input)


# Each layer's runner: run one golden through the layer, raising if it fails.


def run_structured(golden: Golden) -> None:
    # Span metrics are evaluated by `deepeval test run`'s per-test trace
    # machinery; under plain pytest there is no trace to attach them to.
    if current_trace_context.get() is None:
        pytest.skip("no deepeval trace: run via `deepeval test run`, not plain pytest")
    # Runs inside the test's DeepEval trace, so the agent's spans are recorded
    # there. The live agent runs fresh, and fills the answer cache for the
    # later layers.
    if golden.additional_metadata["scenario"] == "pass":
        run_agent(golden.input, fresh=True)
    else:
        replay_tool_calls(golden)
    assert_tool_calls(golden)


def run_fuzzy(golden: Golden) -> None:
    assert_fuzzy_match(golden, answer_for(golden))


def run_judge(golden: Golden) -> None:
    assert_jev_correctness(golden, answer_for(golden))


@dataclass(frozen=True)
class Stage:
    """One layer of evals."""

    name: str
    enabled: bool
    pass_file: str
    fail_file: str
    failure_modes: tuple[str, ...]  # failure modes this layer is expected to catch
    run: Callable[[Golden], None]


# In sequence order: cheapest first, and the judge last as the final arbiter.
# Fuzzy matching checks which concepts an answer covers, not whether what it
# says about them is true, so contradictions are left to the judge.
STAGES = [
    Stage(
        "structured",
        RUN_STRUCTURED_EVALS,
        "agent_structured_pass_goldens.json",
        "agent_structured_fail_goldens.json",
        ("fail_invalid_keywords", "fail_irrelevant_results", "fail_no_tool_call"),
        run_structured,
    ),
    Stage(
        "fuzzy",
        RUN_FUZZY_EVALS,
        # expected_output holds key phrases, one requirement per line.
        "agent_fuzzy_pass_goldens.json",
        "agent_fuzzy_fail_goldens.json",
        ("fail_incomplete",),
        run_fuzzy,
    ),
    Stage(
        "judge",
        RUN_JEV_EVALS,
        # expected_output holds a reference answer.
        "agent_judge_pass_goldens.json",
        "agent_judge_fail_goldens.json",
        ("fail_contradiction", "fail_incomplete"),
        run_judge,
    ),
]
ENABLED_STAGES = [stage for stage in STAGES if stage.enabled]


@dataclass(frozen=True)
class EvalCase:
    stage: Stage
    golden: Golden

    @property
    def query(self) -> str:
        return self.golden.input

    @property
    def scenario(self) -> str:
        return self.golden.additional_metadata["scenario"]


def load_query_groups() -> dict[str, list[EvalCase]]:
    """Each query's eval cases, in sequence order: by layer, then the pass
    golden before the failures. Only the golden sets of enabled layers are loaded."""
    groups: dict[str, list[EvalCase]] = {}
    for stage in ENABLED_STAGES:
        for golden in load_goldens(stage.pass_file) + load_goldens(stage.fail_file):
            groups.setdefault(golden.input, []).append(EvalCase(stage, golden))
    return groups


QUERY_GROUPS = load_query_groups()
# pytest runs parametrised cases in list order, so each query's sequence runs together.
EVAL_CASES = [case for cases in QUERY_GROUPS.values() for case in cases]

# Query -> the layer that failed first for it.
_failed_stages: dict[str, str] = {}


def case_id(case: EvalCase) -> str:
    return f"q{case.golden.additional_metadata['question_id']}-{case.stage.name}-{case.scenario}"


def query_id(query: str) -> str:
    return f"q{QUERY_GROUPS[query][0].golden.additional_metadata['question_id']}"


@pytest.mark.parametrize("query", list(QUERY_GROUPS), ids=query_id)
def test_query_group_complete(query: str):
    """In every enabled layer, the query has one pass golden, and one golden
    per failure mode that layer is expected to catch."""
    scenarios = Counter((case.stage.name, case.scenario) for case in QUERY_GROUPS[query])
    expected = Counter(
        {(stage.name, s): 1 for stage in ENABLED_STAGES for s in ("pass",) + stage.failure_modes}
    )
    assert scenarios == expected


@pytest.mark.parametrize("case", EVAL_CASES, ids=case_id)
def test_eval(case: EvalCase):
    failed_stage = _failed_stages.get(case.query)
    if SEQUENCE_FAIL_FAST and failed_stage and failed_stage != case.stage.name:
        pytest.skip(f"the {failed_stage} layer failed for this query")
    try:
        case.stage.run(case.golden)
    except Exception:
        _failed_stages.setdefault(case.query, case.stage.name)
        raise
