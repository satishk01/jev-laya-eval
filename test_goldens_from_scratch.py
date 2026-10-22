from pathlib import Path

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul, Score
from deepeval.test_case import LLMTestCase, SingleTurnParams

from deepeval_jev_util import JevQuestionMetric, SharedJevDecision

GOLDENS_PATH = Path(__file__).parent / "goldens" / "ai_engineering_goldens.json"

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(str(GOLDENS_PATH))

# (metric name, Jev question). All questions go to Jev in a single call, and
# each one is reported as its own metric.
QUESTIONS = [
    (
        "No contradiction",
        Noul("No fact in actual_output contradicts a fact in expected_output."),
    ),
    (
        "Completeness",
        Score(
            "How much of the detail in expected_output is present in actual_output?",
            levels=["Almost none", "Some", "Most", "All of it"],
        ),
    ),
]

# Whether each metric should pass for each golden scenario. A metric missing
# from a scenario is reported with its score, but doesn't decide pass/fail.
EXPECTED_OUTCOMES = {
    "pass": {"No contradiction": True, "Completeness": True},
    "fail_contradiction": {"No contradiction": False},
    "fail_incomplete": {"No contradiction": True, "Completeness": False},
}


def golden_id(golden: Golden) -> str:
    meta = golden.additional_metadata
    return f"q{meta['question_id']}-{meta['scenario']}"


@pytest.mark.parametrize("golden", dataset.goldens, ids=golden_id)
def test_correctness(golden: Golden):
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=golden.actual_output,
        expected_output=golden.expected_output,
    )
    decision = SharedJevDecision(
        JevEval(
            name="Correctness",
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            questions=[question for _, question in QUESTIONS],
        )
    )
    expected = EXPECTED_OUTCOMES[golden.additional_metadata["scenario"]]
    metrics = [
        JevQuestionMetric(decision, i, name, expected_pass=expected.get(name))
        for i, (name, _) in enumerate(QUESTIONS)
    ]
    assert_test(test_case, metrics)
