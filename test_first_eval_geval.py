# Optional: the same eval with GEval, an LLM-as-judge metric.
# GEval calls an OpenAI model by default, so it needs OPENAI_API_KEY.
# The rest of the series uses JevEval (test_first_eval.py).
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_context_window_answer_geval():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What is a context window?",
        actual_output=(
            "A context window is the maximum number of tokens a model can "
            "process in one request, including the prompt and its reply."
        ),
        expected_output=(
            "A context window is the maximum number of tokens a model can "
            "process in one request. It includes the prompt, conversation "
            "history, retrieved information, tool results, and the model's "
            "reply. If the total exceeds the limit, content must be removed "
            "or the request may fail."
        ),
    )
    assert_test(test_case, [correctness_metric])
