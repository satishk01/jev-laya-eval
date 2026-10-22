"""Generate a small golden set from scratch with DeepEval's Synthesizer.

3 beginner-level AI engineering questions, each expanded into 3 goldens:
1 that should pass and 2 that should fail (9 goldens in total). Each golden's
`additional_metadata` records which scenario it is and the expected verdict.

Load it back with:

    from deepeval.dataset import EvaluationDataset
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file("goldens/ai_engineering_goldens.json")
"""

from typing import List

from pydantic import BaseModel

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.dataset.utils import trimAndLoadJson
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

NUM_QUESTIONS = 3
OUTPUT_DIR = "goldens"
OUTPUT_FILE_NAME = "ai_engineering_goldens"  # no extension: save_as adds .json

styling_config = StylingConfig(
    scenario=(
        "A beginner who is new to AI engineering is asking an assistant to "
        "explain core concepts such as LLMs, tokens, prompts, embeddings, "
        "RAG, temperature, context windows and evaluation."
    ),
    task="Answer beginner-level AI engineering questions accurately and clearly.",
    input_format="A single short question, in plain English, about one AI engineering concept.",
    expected_output_format="A correct, beginner-friendly answer of 2-4 sentences.",
)


class Variants(BaseModel):
    reference: str
    passing: str
    fail_contradiction: str
    fail_incomplete: str


VARIANTS_PROMPT = """You are building test data for evaluating an AI assistant.

Question: {input}

First write "reference": a correct, beginner-friendly reference answer of 2-4 sentences.

Then write three candidate answers to the question:
- "passing": correct, and covers the key points of the reference answer. Reword it; don't copy it.
- "fail_contradiction": sounds confident but states at least one fact that contradicts the reference answer.
- "fail_incomplete": contains no false statements, but is vague and misses most of the detail in the reference answer.

Each answer should be 1-4 sentences. Return JSON only, with the keys "reference", "passing", "fail_contradiction" and "fail_incomplete".
"""


def generate_variants(synthesizer: Synthesizer, golden: Golden) -> Variants:
    # Goldens generated from scratch have no expected_output, so the reference
    # answer is written here alongside the candidate answers.
    prompt = VARIANTS_PROMPT.format(input=golden.input)
    result = synthesizer.model.generate_with_schema(prompt, schema=Variants)
    # Native deepeval models return (output, cost); custom models return output only.
    if synthesizer.using_native_model:
        result, _ = result
    if isinstance(result, Variants):
        return result
    return Variants.model_validate(trimAndLoadJson(result))


def main():
    synthesizer = Synthesizer(styling_config=styling_config)
    base_goldens = synthesizer.generate_goldens_from_scratch(num_goldens=NUM_QUESTIONS)

    goldens: List[Golden] = []
    for question_id, base in enumerate(base_goldens, start=1):
        variants = generate_variants(synthesizer, base)
        scenarios = [
            ("pass", "pass", variants.passing),
            ("fail_contradiction", "fail", variants.fail_contradiction),
            ("fail_incomplete", "fail", variants.fail_incomplete),
        ]
        for scenario, expected_verdict, actual_output in scenarios:
            goldens.append(
                Golden(
                    input=base.input,
                    actual_output=actual_output,
                    expected_output=variants.reference,
                    additional_metadata={
                        "question_id": question_id,
                        "scenario": scenario,
                        "expected_verdict": expected_verdict,
                    },
                )
            )

    dataset = EvaluationDataset(goldens=goldens)
    path = dataset.save_as(
        file_type="json", directory=OUTPUT_DIR, file_name=OUTPUT_FILE_NAME
    )
    print(f"Saved {len(goldens)} goldens to {path}")


if __name__ == "__main__":
    main()
