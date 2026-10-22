"""Generate golden sets for agent.py from the contexts in agent-knowledge.md.

Uses DeepEval's "Generate Goldens From Contexts": each `##` section of the
knowledge base becomes one context, and the Synthesizer writes a question and
reference answer (expected_output) grounded in it. Then, per question, the
synthesizer's model writes two deliberately incorrect answers.

Two files are written:
- goldens/agent_judge_pass_goldens.json: one golden per question, no actual_output.
  The test answers each one live with agent.py and expects it to pass.
- goldens/agent_judge_fail_goldens.json: two goldens per question (fail_contradiction
  and fail_incomplete), with the incorrect answer as actual_output. The test
  replays them through a mocked agent and expects them to fail.

Load either with:

    from deepeval.dataset import EvaluationDataset
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file("goldens/agent_judge_pass_goldens.json")
"""

import re
from collections import Counter
from pathlib import Path
from typing import List

from pydantic import BaseModel

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.dataset.utils import trimAndLoadJson
from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

KNOWLEDGE_PATH = Path(__file__).parent / "agent-knowledge.md"
NUM_CONTEXTS = 8  # sections to sample, evenly spaced through the file; None for all
OUTPUT_DIR = "goldens"
PASS_FILE_NAME = "agent_judge_pass_goldens"  # no extension: save_as adds .json
FAIL_FILE_NAME = "agent_judge_fail_goldens"
FAIL_SCENARIOS = ["fail_contradiction", "fail_incomplete"]

styling_config = StylingConfig(
    scenario="A beginner who is new to AI engineering is asking an assistant about a concept.",
    task="Answer beginner-level AI engineering questions accurately and clearly.",
    input_format="A single short question, in plain English, about one AI engineering concept.",
    expected_output_format="A correct, beginner-friendly answer of 2-4 sentences.",
)

# Only evolutions that stay within the given context. The defaults spread over
# all seven, including Hypothetical and In-Breadth, which drift into questions
# the knowledge base (and so the agent) can't answer.
evolution_config = EvolutionConfig(
    num_evolutions=1,
    evolutions={
        Evolution.REASONING: 1 / 3,
        Evolution.CONCRETIZING: 1 / 3,
        Evolution.COMPARATIVE: 1 / 3,
    },
)


def load_contexts(path: Path) -> List[tuple[str, List[str]]]:
    """One (section heading, paragraphs) context per `##` section."""
    sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]
    contexts = []
    for section in sections:
        heading, _, body = section.partition("\n")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        contexts.append((heading.strip(), paragraphs))
    return contexts


def sample_evenly(items: list, count) -> list:
    if count is None or count >= len(items):
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


class FailureVariants(BaseModel):
    fail_contradiction: str
    fail_incomplete: str


FAILURES_PROMPT = """You are building test data for evaluating an AI assistant.

Question: {input}
Reference answer: {expected_output}

Write two deliberately incorrect answers to the question:
- "fail_contradiction": sounds confident but states at least one fact that contradicts the reference answer.
- "fail_incomplete": contains no false statements, but is vague and misses most of the detail in the reference answer.

Each answer should be 1-4 sentences. Return JSON only, with the keys "fail_contradiction" and "fail_incomplete".
"""


def generate_failures(synthesizer: Synthesizer, golden: Golden) -> FailureVariants:
    prompt = FAILURES_PROMPT.format(input=golden.input, expected_output=golden.expected_output)
    result = synthesizer.model.generate_with_schema(prompt, schema=FailureVariants)
    # Native deepeval models return (output, cost); custom models return output only.
    if synthesizer.using_native_model:
        result, _ = result
    if isinstance(result, FailureVariants):
        return result
    return FailureVariants.model_validate(trimAndLoadJson(result))


def check_alignment(pass_goldens: List[Golden], fail_goldens: List[Golden]) -> None:
    """Every pass query must appear exactly once per failure scenario in the fail set."""
    pass_inputs = Counter(g.input for g in pass_goldens)
    fail_pairs = Counter((g.input, g.additional_metadata["scenario"]) for g in fail_goldens)
    expected_pairs = Counter({(i, s): 1 for i in pass_inputs for s in FAIL_SCENARIOS})
    if any(n != 1 for n in pass_inputs.values()) or fail_pairs != expected_pairs:
        raise ValueError("Pass and fail golden sets are not aligned.")


def main():
    contexts = sample_evenly(load_contexts(KNOWLEDGE_PATH), NUM_CONTEXTS)
    synthesizer = Synthesizer(styling_config=styling_config, evolution_config=evolution_config)
    base_goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[paragraphs for _, paragraphs in contexts],
        include_expected_output=True,
        max_goldens_per_context=1,
    )

    pass_goldens: List[Golden] = []
    fail_goldens: List[Golden] = []
    for question_id, base in enumerate(base_goldens, start=1):
        if not base.expected_output:
            raise ValueError(f"Golden has no expected_output: {base.input!r}")
        # Each golden carries its context, so match it back to the section heading.
        section = next(heading for heading, paragraphs in contexts if paragraphs == base.context)
        metadata = {"question_id": question_id, "section": section}

        pass_goldens.append(
            Golden(
                input=base.input,
                expected_output=base.expected_output,
                context=base.context,
                additional_metadata={**metadata, "scenario": "pass", "expected_verdict": "pass"},
            )
        )
        failures = generate_failures(synthesizer, base)
        for scenario in FAIL_SCENARIOS:
            fail_goldens.append(
                Golden(
                    input=base.input,
                    actual_output=getattr(failures, scenario),
                    expected_output=base.expected_output,
                    context=base.context,
                    additional_metadata={**metadata, "scenario": scenario, "expected_verdict": "fail"},
                )
            )

    check_alignment(pass_goldens, fail_goldens)
    for goldens, file_name in [(pass_goldens, PASS_FILE_NAME), (fail_goldens, FAIL_FILE_NAME)]:
        path = EvaluationDataset(goldens=goldens).save_as(
            file_type="json", directory=OUTPUT_DIR, file_name=file_name
        )
        print(f"Saved {len(goldens)} goldens to {path}")


if __name__ == "__main__":
    main()
