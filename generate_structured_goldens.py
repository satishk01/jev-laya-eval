"""Generate golden sets for structured validation of agent.py's tool calls.

Same queries as goldens/agent_judge_pass_goldens.json. No LLM is needed: the
mocked tool calls are built from each query's knowledge base section heading
and the real search tool.

- goldens/agent_structured_pass_goldens.json: one golden per query, with
  expected_tools = [search_knowledge]. tools_called is left empty: the eval
  records it from the live agent's tool spans.
- goldens/agent_structured_fail_goldens.json: one golden per query and failure
  mode, with tools_called holding the tool calls a mocked agent replays:
  - fail_invalid_keywords: keywords break the tool's contract (a multi-word
    keyword, and more than 5 keywords).
  - fail_irrelevant_results: valid keywords, but the output includes a
    paragraph containing none of them.
  - fail_no_tool_call: the agent answers without searching. deepeval saves an
    empty tools_called list as null, so here null means "no tool calls".
"""

import re
from pathlib import Path
from typing import List

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

import agent

GOLDENS_DIR = Path(__file__).parent / "goldens"
SOURCE_FILE = GOLDENS_DIR / "agent_judge_pass_goldens.json"
PASS_FILE_NAME = "agent_structured_pass_goldens"  # no extension: save_as adds .json
FAIL_FILE_NAME = "agent_structured_fail_goldens"
FAILURE_MODES = ["fail_invalid_keywords", "fail_irrelevant_results", "fail_no_tool_call"]
TOOL_NAME = "search_knowledge"
STOP_WORDS = {"and", "the", "of", "vs", "a", "an", "to", "in", "is"}

# The tool without its @observe wrapper, so generating goldens records no traces.
search_knowledge = agent.search_knowledge.__wrapped__


def valid_keywords(section: str) -> List[str]:
    """Up to 5 single-word keywords taken from the section heading."""
    words = re.sub(r"[^\w\s-]", " ", section.lower()).split()
    return [w for w in words if w not in STOP_WORDS][:5]


def invalid_keywords(section: str) -> List[str]:
    """Breaks the contract twice: the whole heading as one multi-word keyword,
    and 6 keywords in total."""
    padding = ["ai", "engineering", "basics", "concept", "definition", "example"]
    keywords = [section.lower()] + valid_keywords(section)
    return (keywords + padding)[:6]


def with_irrelevant_result(output: str, keywords: List[str]) -> str:
    """The real search output, with its last result swapped for a paragraph
    (and its headings) that contains none of the keywords."""
    patterns = [re.compile(r"\b" + re.escape(k), re.IGNORECASE) for k in keywords]
    headings, text = next(
        (h, t)
        for h, t in agent.load_paragraphs(agent.KNOWLEDGE_PATH)
        if not any(p.search(t) for p in patterns)
    )
    summary, *results = output.split("\n\n---\n\n")
    if not results:
        raise ValueError(f"No results to replace for keywords {keywords}")
    results[-1] = "\n".join(headings + [text])
    return "\n\n---\n\n".join([summary] + results)


def tool_call(keywords: List[str], output: str) -> ToolCall:
    return ToolCall(name=TOOL_NAME, input_parameters={"keywords": keywords}, output=output)


def failure_tools_called(scenario: str, section: str) -> List[ToolCall]:
    if scenario == "fail_invalid_keywords":
        keywords = invalid_keywords(section)
        return [tool_call(keywords, search_knowledge(keywords))]
    if scenario == "fail_irrelevant_results":
        keywords = valid_keywords(section)
        return [tool_call(keywords, with_irrelevant_result(search_knowledge(keywords), keywords))]
    if scenario == "fail_no_tool_call":
        return []
    raise ValueError(f"Unknown failure mode: {scenario}")


def load_goldens(path: Path) -> List[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(path))
    return dataset.goldens


def main():
    expected_tools = [ToolCall(name=TOOL_NAME)]
    pass_goldens: List[Golden] = []
    fail_goldens: List[Golden] = []
    for source in load_goldens(SOURCE_FILE):
        meta = source.additional_metadata
        base = {"question_id": meta["question_id"], "section": meta["section"]}
        pass_goldens.append(
            Golden(
                input=source.input,
                expected_tools=expected_tools,
                additional_metadata={**base, "scenario": "pass", "expected_verdict": "pass"},
            )
        )
        for scenario in FAILURE_MODES:
            fail_goldens.append(
                Golden(
                    input=source.input,
                    tools_called=failure_tools_called(scenario, meta["section"]),
                    expected_tools=expected_tools,
                    additional_metadata={**base, "scenario": scenario, "expected_verdict": "fail"},
                )
            )

    for goldens, file_name in [(pass_goldens, PASS_FILE_NAME), (fail_goldens, FAIL_FILE_NAME)]:
        path = EvaluationDataset(goldens=goldens).save_as(
            file_type="json", directory=str(GOLDENS_DIR), file_name=file_name
        )
        print(f"Saved {len(goldens)} goldens to {path}")


if __name__ == "__main__":
    main()
