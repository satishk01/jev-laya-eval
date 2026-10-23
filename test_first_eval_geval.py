# Optional: the same eval with GEval, an LLM-as-judge metric.
# GEval calls an OpenAI model by default, so it needs OPENAI_API_KEY.
# The rest of the series uses JevEval (test_first_eval.py).
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()


def _judge_model_configured() -> bool:
    """Whether GEval can resolve a generative judge: a real OPENAI_API_KEY,
    LOCAL_MODEL_*/OLLAMA_* env vars, or a provider persisted by `deepeval
    set-*` into .deepeval/.deepeval. A placeholder key still counts as
    unconfigured -- the run would fail on auth anyway."""
    if (os.environ.get("OPENAI_API_KEY") or "").startswith("sk-"):
        return True
    provider_keys = (
        "USE_LOCAL_MODEL",
        "LOCAL_MODEL_NAME",
        "LOCAL_MODEL_BASE_URL",
        "OLLAMA_MODEL_NAME",
    )
    if any(os.environ.get(k) for k in provider_keys):
        return True
    config = Path(".deepeval") / ".deepeval"
    if config.is_file():
        try:
            data = json.loads(config.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return any(data.get(k) for k in provider_keys + ("LOCAL_MODEL_API_KEY",))
    return False


pytestmark = pytest.mark.skipif(
    not _judge_model_configured(),
    reason="GEval needs an LLM judge: set a real OPENAI_API_KEY, or configure a "
    "local model (deepeval set-ollama / LOCAL_MODEL_*)",
)


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
