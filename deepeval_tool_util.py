"""Structured validation of agent.py's tool calls, as a deepeval span metric.

Rather than judging the agent's final answer, this checks each
`search_knowledge` call recorded as an `@observe(type="tool")` span:

- input (the keywords the model sent, as JSON) follows the tool's contract:
  1-5 keywords, each a single non-empty word.
- output is well formed: either a "no match" message, or a summary line
  "N paragraphs matched; showing M." followed by exactly M results, with
  M <= N and M <= the result limit; each result is its heading lines ("#...")
  followed by the paragraph text.
- every returned paragraph contains at least one of the keywords.

It also copies the tool spans into the trace's `tools_called`, so deepeval's
ToolCorrectnessMetric can compare them with a golden's `expected_tools`.

deepeval builds the span's test case with input=str(span.input) and
actual_output=str(span.output), so the metric parses those strings.
"""

import json
import re
from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import BaseSpan, ToolSpan

RESULT_SEPARATOR = "\n\n---\n\n"
SUMMARY_PATTERN = re.compile(r"(\d+) paragraphs matched; showing (\d+)\.")
NO_MATCH_PREFIXES = ("No paragraphs matched:", "No keywords given.")
MAX_KEYWORDS = 5


def validate_input(raw_input: str) -> List[str]:
    """Problems with the tool call's input; an empty list means it's valid."""
    try:
        keywords = json.loads(raw_input).get("keywords")
    except (json.JSONDecodeError, AttributeError):
        return [f"input is not a JSON object: {raw_input[:80]!r}"]
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        return ["keywords is not a list of strings"]

    problems = []
    if not 1 <= len(keywords) <= MAX_KEYWORDS:
        problems.append(f"{len(keywords)} keywords, expected 1-{MAX_KEYWORDS}")
    if any(not k.strip() for k in keywords):
        problems.append("empty keyword")
    multi_word = [k for k in keywords if len(k.split()) > 1]
    if multi_word:
        problems.append(f"multi-word keywords {multi_word}, expected single words")
    return problems


def search_words(raw_input: str) -> List[str]:
    """The words the tool actually searched for: phrases split into words."""
    try:
        keywords = json.loads(raw_input).get("keywords") or []
    except (json.JSONDecodeError, AttributeError):
        return []
    return [word for k in keywords if isinstance(k, str) for word in k.split()]


def validate_output(raw_output: str, words: List[str], max_results: int) -> List[str]:
    """Problems with the tool call's output; an empty list means it's valid."""
    if raw_output.startswith(NO_MATCH_PREFIXES):
        return []

    summary, *results = raw_output.split(RESULT_SEPARATOR)
    match = SUMMARY_PATTERN.fullmatch(summary.strip())
    if not match:
        return [f"unrecognised output: {summary[:80]!r}"]

    matched, shown = int(match.group(1)), int(match.group(2))
    problems = []
    if shown > matched:
        problems.append(f"shows {shown} of only {matched} matches")
    if shown > max_results:
        problems.append(f"shows {shown} results, limit is {max_results}")
    if len(results) != shown:
        problems.append(f"summary says {shown} results, output has {len(results)}")

    patterns = [re.compile(r"\b" + re.escape(w), re.IGNORECASE) for w in words]
    for i, result in enumerate(results, start=1):
        lines = result.split("\n")
        headings = [line for line in lines if line.startswith("#")]
        paragraph = "\n".join(line for line in lines if not line.startswith("#")).strip()
        if not headings or lines[: len(headings)] != headings:
            problems.append(f"result {i} doesn't start with its headings")
        if not paragraph:
            problems.append(f"result {i} has no paragraph text")
        elif patterns and not any(p.search(paragraph) for p in patterns):
            problems.append(f"result {i} contains none of the keywords")
    return problems


class ToolCallValidationMetric(BaseMetric):
    """Pass/fail structural validation of one search_knowledge tool call.

    Succeeds when the verdict (valid or not) matches `expected_pass`; with
    `expected_pass=None` it has no pass/fail verdict.
    """

    def __init__(
        self,
        max_results: int,
        expected_pass: Optional[bool] = True,
        name: str = "Tool call validation",
    ):
        self.max_results = max_results
        self.expected_pass = expected_pass
        self.name = name
        self.threshold = 1.0  # every check must pass
        self.include_reason = True
        self.evaluation_model = "rules"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raw_input = test_case.input or ""
        raw_output = test_case.actual_output or ""
        problems = validate_input(raw_input) + validate_output(
            raw_output, search_words(raw_input), self.max_results
        )

        self.score = 0.0 if problems else 1.0
        self.evaluation_cost = 0
        self.success = self.is_successful()
        expected = {True: "valid", False: "invalid", None: "either"}[self.expected_pass]
        self.reason = (
            f"Invalid tool call {raw_input}, expected {expected}: " + "; ".join(problems)
            if problems
            else f"Valid tool call {raw_input}, expected {expected}: "
            f"{raw_output.splitlines()[0] if raw_output else ''}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> Optional[bool]:
        if self.error is not None:
            self.success = False
        elif self.expected_pass is None:
            self.success = None
        else:
            valid = self.score is not None and self.score >= self.threshold
            self.success = valid == self.expected_pass
        return self.success

    @property
    def __name__(self):
        return self.name


class ExpectedVerdictMetric(BaseMetric):
    """Wraps any metric so it succeeds when the wrapped metric's verdict matches
    `expected_pass`, e.g. ToolCorrectnessMetric correctly failing a mocked
    agent that never called its tool. With `expected_pass=None` it has no
    pass/fail verdict. An error in the wrapped metric is always a failure."""

    def __init__(self, metric: BaseMetric, expected_pass: Optional[bool], name: Optional[str] = None):
        self.metric = metric
        self.expected_pass = expected_pass
        self.name = name or metric.__name__
        self.threshold = metric.threshold
        self.include_reason = True
        self.evaluation_model = metric.evaluation_model

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.metric.measure(test_case, *args, **kwargs)
        self._finalize()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        await self.metric.a_measure(test_case, *args, **kwargs)
        self._finalize()
        return self.score

    def _finalize(self) -> None:
        self.score = self.metric.score
        self.error = self.metric.error
        self.evaluation_cost = self.metric.evaluation_cost
        self.success = self.is_successful()
        verdict = "pass" if self.metric.success else "fail"
        expected = {True: "pass", False: "fail", None: "either"}[self.expected_pass]
        self.reason = f"{verdict}, expected {expected}. {self.metric.reason}"

    def is_successful(self) -> Optional[bool]:
        if self.error is not None:
            self.success = False
        elif self.expected_pass is None:
            self.success = None
        else:
            self.success = bool(self.metric.success) == self.expected_pass
        return self.success

    @property
    def __name__(self):
        return self.name


def tool_spans() -> List[ToolSpan]:
    """All tool spans in the current deepeval trace (the running test's trace)."""
    trace = current_trace_context.get()
    spans: List[BaseSpan] = list(trace.root_spans or []) if trace else []
    found = []
    while spans:
        span = spans.pop(0)
        if isinstance(span, ToolSpan):
            found.append(span)
        spans.extend(span.children or [])
    return found


def attach_tool_metrics(max_results: int, expected_pass: Optional[bool] = True) -> int:
    """Attach a fresh ToolCallValidationMetric to every tool span in the
    current trace, so `assert_test(golden=...)` evaluates them. Returns the
    number of tool spans found."""
    spans = tool_spans()
    for span in spans:
        span.metrics = [ToolCallValidationMetric(max_results=max_results, expected_pass=expected_pass)]
    return len(spans)


def record_tools_called() -> List[ToolCall]:
    """Set the current trace's `tools_called` from its tool spans, which is
    where trace-level metrics like ToolCorrectnessMetric read them."""
    calls = []
    for span in tool_spans():
        try:
            input_parameters = json.loads(span.input) if isinstance(span.input, str) else span.input
        except json.JSONDecodeError:
            input_parameters = {"raw": span.input}
        calls.append(ToolCall(name=span.name, input_parameters=input_parameters, output=span.output))
    current_trace_context.get().tools_called = calls
    return calls
