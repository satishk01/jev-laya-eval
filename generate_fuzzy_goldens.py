"""Derive fuzzy-matching golden sets from the judge golden sets.

Same queries and same deliberately wrong answers as
goldens/agent_judge_{pass,fail}_goldens.json, except for fail_contradiction:
keyword matching checks which concepts an answer covers, not whether what it
says about them is true, so contradictions are left to the Jev evals. And
expected_output holds the
key phrase requirements a correct answer must meet instead of a full reference
answer: one per line, with alternative phrasings separated by " | " (see
deepeval_fuzzy_util). An LLM extracts them from each reference answer.

Writes goldens/agent_fuzzy_pass_goldens.json and goldens/agent_fuzzy_fail_goldens.json.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.dataset.utils import trimAndLoadJson
from deepeval.models import GPTModel

from deepeval_fuzzy_util import ALTERNATIVE_SEPARATOR, format_key_phrases

GOLDENS_DIR = Path(__file__).parent / "goldens"
JUDGE_PASS_FILE = GOLDENS_DIR / "agent_judge_pass_goldens.json"
JUDGE_FAIL_FILE = GOLDENS_DIR / "agent_judge_fail_goldens.json"
PASS_FILE_NAME = "agent_fuzzy_pass_goldens"  # no extension: save_as adds .json
FAIL_FILE_NAME = "agent_fuzzy_fail_goldens"
FAILURE_MODES = ["fail_incomplete"]  # failure modes fuzzy matching is expected to catch


class KeyPhrases(BaseModel):
    key_phrases: List[List[str]]  # requirements, each a list of alternative phrasings


KEY_PHRASES_PROMPT = """You are building test data for checking an AI assistant's answers by keyword matching.

Question: {input}
Reference answer: {expected_output}

List the 3-5 key concepts that any correct answer to this question must mention.
Each concept is a list of 1-4 alternative phrasings, any one of which counts:
- The first phrasing is copied from the reference answer.
- Add common abbreviations, synonyms and spelling variants that a correct answer
  might use instead, e.g. ["personally identifiable information", "PII"] or
  ["nearest neighbour", "nearest neighbor", "ANN"].
- Each phrasing is 1-3 words. Use lowercase, except for abbreviations.
- Prefer central concepts and facts over incidental details or generic words like "model", "data" or "important".
- Don't include concepts that merely repeat the question's wording.

Return JSON only, with the key "key_phrases".
"""


def extract_key_phrases(model: GPTModel, golden: Golden) -> List[List[str]]:
    prompt = KEY_PHRASES_PROMPT.format(input=golden.input, expected_output=golden.expected_output)
    result, _ = model.generate(prompt, schema=KeyPhrases)
    if not isinstance(result, KeyPhrases):
        result = KeyPhrases.model_validate(trimAndLoadJson(result))
    # The separator can't appear inside a phrasing.
    requirements = [
        [p.replace(ALTERNATIVE_SEPARATOR, " ").strip() for p in alternatives if p.strip()]
        for alternatives in result.key_phrases
    ]
    requirements = [alternatives for alternatives in requirements if alternatives]
    if not requirements:
        raise ValueError(f"No key phrases extracted for: {golden.input!r}")
    return requirements


def load_goldens(path: Path) -> List[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(path))
    return dataset.goldens


def with_key_phrases(golden: Golden, key_phrases: List[List[str]]) -> Golden:
    return golden.model_copy(update={"expected_output": format_key_phrases(key_phrases)})


def main():
    model = GPTModel()
    judge_pass = load_goldens(JUDGE_PASS_FILE)
    judge_fail = load_goldens(JUDGE_FAIL_FILE)

    key_phrases = {golden.input: extract_key_phrases(model, golden) for golden in judge_pass}
    fuzzy_pass = [with_key_phrases(g, key_phrases[g.input]) for g in judge_pass]
    fuzzy_fail = [
        with_key_phrases(g, key_phrases[g.input])
        for g in judge_fail
        if g.additional_metadata["scenario"] in FAILURE_MODES
    ]

    for goldens, file_name in [(fuzzy_pass, PASS_FILE_NAME), (fuzzy_fail, FAIL_FILE_NAME)]:
        path = EvaluationDataset(goldens=goldens).save_as(
            file_type="json", directory=str(GOLDENS_DIR), file_name=file_name
        )
        print(f"Saved {len(goldens)} goldens to {path}")


if __name__ == "__main__":
    main()
