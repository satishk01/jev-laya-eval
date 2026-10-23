# Evaluating AI Agents with Jev and Laya: A Technical Deep Dive

**Audience:** engineers already comfortable with LLM application basics (prompts,
tools, RAG) who want to build serious evaluation pipelines — roughly AWS "level
200" material: working code, architecture trade-offs, and production lessons, not
a product overview.

**Scope:** this document describes the evaluation architecture in this
repository — a three-layer eval harness for a ReAct agent, where the top layer is
a *decision-model judge* that can run either as a hosted service (**Jev**) or as
a local open-weight checkpoint (**Laya**) behind the same wire protocol.

---

## 1. The problem: evaluating agents is harder than evaluating models

A single-turn LLM eval asks: "is this output good?" An agent eval asks harder
questions:

- **Did it use its tools correctly?** A RAG-style agent can produce a fluent,
correct-sounding answer while its retrieval calls were malformed, returned
nothing, or were never made at all.
- **Does the answer cover what it should?** An answer can be entirely true and
still useless because it omits most of the required detail.
- **Is what it says actually true?** Coverage checks can't tell whether the
claims made are supported by the reference material.

No single metric answers all three. Keyword matching can't judge truth; a judge
model can't cheaply verify JSON contracts; and running an LLM judge over every
assertion is slow and expensive. The design that falls out of this is a
**layered eval**: cheap deterministic checks first, expensive semantic checks
last, with each layer catching a distinct class of failure.

This repository implements exactly that for `agent.py`, a minimal ReAct agent
that answers beginner AI-engineering questions by searching a Markdown knowledge
base (`agent-knowledge.md`) with one default tool, `search_knowledge(keywords)`.
An optional second tool, `list_sections()` — which returns the knowledge base's
section outline for meta-questions ("What topics do you cover?") — is gated
behind `AGENT_TOOLS=search_knowledge,list_sections`; it exists so the extended
suite can test real multi-tool selection without changing default behaviour.

| Layer | Cost per test | What it catches | Implementation |
|---|---|---|---|
| 1. Structured | ~0 (rules) | malformed tool calls, missing tool calls | `deepeval_tool_util.py` + `ToolCorrectnessMetric` |
| 2. Fuzzy | ~0 (local NLP) | thin/incomplete answers | `deepeval_fuzzy_util.py` (exact → RapidFuzz → BLEU/ROUGE) |
| 3. Judge | one decision-model call per test | contradictions vs. a reference answer | `JevEval` → **Jev** (hosted) or **Laya** (local) |

Layers run cheapest-first per query, and `SEQUENCE_FAIL_FAST` skips a query's
later layers once an early layer fails — a broken tool call shouldn't burn judge
calls.

---

## 2. The judge: System-1 decision models (Jev and Laya)

### 2.1 What a decision model is

Most "LLM-as-judge" setups prompt a general-purpose generative model ("You are
an impartial evaluator… rate 1–5…") and parse prose back into a score. That
approach has well-known problems: position bias, verbose hedging, inconsistent
calibration, sensitivity to prompt phrasing, and per-token cost for what is
fundamentally a classification task.

**Jev** and **Laya** are a different kind of model: *System-1 decision models*.
Instead of generating a judgement as text, they score a set of **states** (the
facts of the test case: input, actual output, expected output, context) against
a set of explicit **questions**, and return a calibrated probability per
question. You ask questions; the model returns probabilities. No prose to parse,
no rubric drift between runs.

The questions come in three types:

| Question type | Shape | Returned value | Use in this repo |
|---|---|---|---|
| `Choice` | A statement + a set of labelled options, each with a credit | Probability-weighted value over the options | Per-sentence contradiction checks: `{"agrees": 1.0, "contradicts": 0.0}` → P(sentence agrees) |
| `Score` | A question + ordered rubric levels | Value in [0, 1] over the levels | Completeness: "How much of expected_output is present in actual_output?" with levels `["Almost none", "Some", "Most", "All of it"]` |
| `Noul` | A yes/no statement | P(statement holds) | Available in the protocol but **avoided** here — see §6.3 |

A `Choice` over `{agrees: 1.0, contradicts: 0.0}` is numerically equivalent to a
noul — its expected value *is* the probability the statement holds — which is
why this codebase uses Choices as its binary-verdict workhorse.

### 2.2 Jev: the hosted judge

**Jev** is the hosted System-1 decision service, reached through the **TypeSafe
SDK** (`typesafe-sdk` on PyPI). DeepEval wraps it as the `JevEval` metric
(`deepeval.metrics.JevEval`). Configuration is three environment variables:

```ini
TYPESAFE_BASE_URL=<your gateway>
TYPESAFE_API_KEY=<your key>
TYPESAFE_MODEL_NAME=jev-1.13
```

Jev-1.13 is a strong general judge: in this project it reliably caught both
contradictions *and* under-complete answers. The cost is a network dependency,
API keys, and per-request billing.

### 2.3 Laya: the open-weight local judge

**Laya** (`convaiinnovations/laya`) is the open-weight release of the same
System-1 decision-model family. The `laya[serve]` extra installs `laya-serve`,
a local inference server that **speaks Jev's wire protocol** —
`POST /v1/systemone` — so DeepEval's `JevEval` and the TypeSafe SDK work against
it unchanged. Checkpoint variants:

- `typed-decisions` — fine-tuned for judge-type questions; the recommended one
  and the one used throughout this repo
- `english`, `multilingual` — general variants
- a Jev-style id such as `jev-1` — falls through to Laya's router

`run_judge.py` is a thin launcher:

```python
os.environ.setdefault("LAYA_HOST", "127.0.0.1")
os.environ.setdefault("LAYA_PORT", "8000")
os.environ.setdefault("LAYA_PRELOAD", "0")   # lazy-load checkpoints

from laya.serve import main
```

Running on CPU, a loaded checkpoint answers a decision request in roughly
0.2–0.5s per question pass (1–3s per multi-question call in this suite's
observed runs). The first request after startup can take a minute or two while
the checkpoint downloads and loads — `/health` reports a `loaded` list; wait
for the checkpoint to appear there before running tests.

### 2.4 One protocol, two backends — the key architectural property

The important design point: **which judge answers is decided entirely by
environment variables, not by code.** The eval suite, the metrics, and the
question definitions are identical either way:

```ini
# Local Laya
TYPESAFE_BASE_URL=http://localhost:8000   # SDK appends /v1/systemone itself
TYPESAFE_API_KEY=local-laya               # any non-empty string
TYPESAFE_MODEL_NAME=typed-decisions

# Hosted Jev
TYPESAFE_BASE_URL=<your gateway>
TYPESAFE_API_KEY=<your key>
TYPESAFE_MODEL_NAME=jev-1.13
```

In practice this means you can develop the whole eval stack offline with zero
API keys (Laya + the agent's deterministic `extractive` backend), then flip to
the hosted judge — or vice versa — by editing `.env` alone. It also means every
quirk of the *checkpoint* is visible through the same code path, which is how
the behavioural differences in §6 were found and worked around.

```
DeepEval JevEval ──► TypeSafe SDK ──► POST /v1/systemone ──► ┌────────────────┐
                                                            │ Hosted Jev     │  (jev-1.13)
                                                            │   or           │
                                                            │ laya-serve     │  (typed-decisions…)
                                                            └────────────────┘
```

---

## 3. How DeepEval's JevEval works

`JevEval` is a DeepEval `BaseMetric` whose `measure()` call packages the test
case's selected fields (`evaluation_params`, e.g. `ACTUAL_OUTPUT` and
`EXPECTED_OUTPUT`) into *states*, sends them with your `questions` to
`/v1/systemone`, and stores per-question outcomes. The metric score is a
weighted aggregate of outcome values; each outcome carries the question text
and the full probability distribution over its options/levels.

This repo builds two pieces of machinery on top of it (`deepeval_jev_util.py`):

### 3.1 `SharedJevDecision` — one judge call, many metrics

A natural instinct is one `JevEval` per assertion. But a judge call is the
expensive unit: every call re-encodes the states and re-runs the model. When
several metrics (e.g. "No contradiction", "Completeness") evaluate the *same*
test case, they should share the call.

`SharedJevDecision` wraps a `JevEval`, keyed by
`(input, actual_output, expected_output)`. The first metric to measure a given
test case makes the request; subsequent metrics for the same case reuse the
cached outcomes. An `asyncio.Lock` serialises the async path so concurrent
DeepEval metric evaluation doesn't issue duplicate calls — the second metric
waits on the first's lock instead of re-deciding.

### 3.2 `JevQuestionMetric` — slicing shared outcomes into named metrics

`JevQuestionMetric` takes one or more `SharedJevDecision`s plus the indices of
the questions it owns, and reports their **minimum** value as its score. Min —
not mean — is the important choice for a contradiction check: a 5-sentence
answer with 4 perfect sentences and 1 false claim should fail, and `min` is the
aggregator that guarantees one bad sentence sinks the whole answer (the standard
decompose-and-verify pattern).

Two supporting details worth copying:

- **Expected-verdict semantics.** Every metric takes `expected_pass`. A test for
  a *deliberately broken* golden succeeds only if the metric *fails* — so
  `is_successful()` compares the threshold verdict to `expected_pass`, and
  `None` means "report the score, don't gate on it." This is what lets one suite
  assert both "the agent passes" and "the harness catches failure X."
- **Cost attribution.** Only the `JevQuestionMetric` instance that actually
  issued the Jev call carries `evaluation_cost`; the sharers report 0, so run
  totals aren't double-counted.
- **Honest brittleness.** `_finalize` reads `jev._system_one_outcomes`, a
  private attribute — flagged in code as a non-public API. If you lift this
  pattern, pin your `deepeval` version.

### 3.3 Chunking questions across calls

Each question in a call costs the model a pass over the encoded states. A
6-sentence answer produces 7 questions (6 contradiction Choices + 1 completeness
Score), and on a CPU-resident checkpoint a single call carrying all of them can
exceed the request timeout. `QUESTIONS_PER_CALL = 4` splits the question list
across multiple `JevEval`/`SharedJevDecision` pairs; `JevQuestionMetric` accepts
a list of decisions and indexes into their concatenated outcomes, so the
chunking is invisible to the test.

---

## 4. The three layers in detail

### 4.1 Structured layer: validating tool calls from traces

The agent's LLM calls, tool calls, and agent loop are decorated with
`@observe`, so every test that runs the agent produces a DeepEval **trace** of
spans. The structured layer inspects the `search_knowledge` tool spans, not the
answer text:

- **Input contract** (`validate_input`): `keywords` must be a JSON array of 1–5
  non-empty *single-word* strings — the same contract `agent.py`'s
  `STEP_SCHEMA` enforces during decoding.
- **Output contract** (`validate_output`): either a recognised "no match"
  message, or the summary line `N paragraphs matched; showing M.` followed by
  exactly M result blocks, with `M ≤ N` and `M ≤ agent.MAX_SEARCH_RESULTS`.
- **Semantic sanity**: every returned paragraph must contain at least one
  search keyword; each result block must lead with its Markdown heading lines.

`attach_tool_metrics` hangs a fresh `ToolCallValidationMetric` on every tool
span in the current test's trace, so `assert_test(golden=...)` evaluates each
call separately. It takes a `tool_name` parameter (default `search_knowledge`)
and only attaches the keyword-contract validator to spans of that tool —
other tools' spans (e.g. `list_sections`) are left for
`ToolCorrectnessMetric`. `record_tools_called` then copies the spans into the
trace's `tools_called`, where `ToolCorrectnessMetric` compares them against the
golden's `expected_tools` — catching the "never called the tool" failure that
per-call validation can't see (an invalid call still *was* a call).

The failure-mode goldens exercise each hole: `fail_invalid_keywords`,
`fail_irrelevant_results`, `fail_no_tool_call` — the first two expect validation
to fail while correctness still "passes" (a call was made), the third expects
the inverse.

### 4.2 Fuzzy layer: key-phrase coverage without a model

The fuzzy layer reframes `expected_output`: instead of a reference answer it
holds **requirements**, one per line, each with alternative phrasings separated
by `|`:

```
personally identifiable information | personal information | PII
identify a person | identifiable data | identity-revealing
```

`FuzzyMatchMetric` scores the fraction of requirements met, trying matchers in
escalating cost order per alternative:

1. **Exact** — normalized whole-word containment.
2. **RapidFuzz** — best `fuzz.ratio` over same-length word windows of the
   answer, cutoff 85; catches typos and inflections. Skipped for phrases under
   5 characters (one-letter edits dominate tiny strings: "rag" vs "drag" ≈ 86).
3. **N-gram** — BLEU (max order 2) and ROUGE-1/2 recall over Porter-stemmed
   tokens, cutoff 0.6; catches reordering and different word forms.

Threshold 0.6 of requirements. This layer is where **incomplete** answers are
gated: a vague but truthful answer fails phrase coverage cheaply, without
spending a judge call. That division of labour is deliberate — see §6.4 for why
the local judge can't be trusted with coverage.

### 4.3 Judge layer: per-sentence verification against a reference

For each answer, `assert_jev_correctness` builds:

- one `Choice("Does this statement agree with expected_output: \"{sentence}\"",
  {"agrees": 1.0, "contradicts": 0.0})` **per sentence** of `actual_output`, and
- one `Score` completeness question.

The questions are chunked into `JevEval` calls of 4, wrapped in
`SharedJevDecision`s, then surfaced as two `JevQuestionMetric`s:

- **"No contradiction"** = `min` over every sentence's agree-probability.
- **"Completeness"** = the single Score outcome.

Checking *sentences* rather than the whole answer is not pedantry — measured
behaviour: the `typed-decisions` checkpoint scores a fluent answer containing
one blatant false claim at ~0.7 "agrees" when asked once about the whole answer,
but flags the offending sentence reliably when it's asked about on its own.
Decompose, verify each part, take the worst.

### 4.4 Fail-fast sequencing

`STAGES` declares each layer with its golden files and the failure modes it
owns; `load_query_groups` groups all `EvalCase`s by query in stage order, and
`test_eval` records the first failing stage per query in `_failed_stages`. With
`SEQUENCE_FAIL_FAST = True`, later layers for that query `pytest.skip` rather
than running — the report shows skips downstream of a real failure, which is
what you want: a tool-contract violation already tells you the answer will be
garbage.

The full suite is 72 eval cases across 8 queries (80 pytest items
including the 8 `test_query_group_complete` data checks; ~3–4 min — the
judge phase dominates). Results land in `deepeval_results/test_run_*.json`.

A second file, `test_agent_extended.py`, layers behavioural checks on top:
trajectory/step budgets and keyword relevance from the trace, the refusal
path, faithfulness of each answer sentence against the retrieved
paragraphs (`GroundingMetric`), multi-turn history bounds, answer
stability across repeated runs, and — when `AGENT_TOOLS` enables
`list_sections` — real multi-tool selection. 62 pytest items: 50 counted
evals with the default single tool (54 with `list_sections` enabled), 6
plain-assert checks (multi-turn + extractive routing), 2 `llm`-gated
tests. Its trace-reading tests also work under plain `pytest` — they start
a fresh `deepeval.tracing.trace()` via `ensure_trace()` when the test-run
plugin hasn't provided one; the structured layer in `test_agent.py` can't
do that (its span metrics are evaluated by the plugin) and skips cleanly
instead.

---

## 5. Goldens: manufacturing both sides of every assertion

Every eval layer consumes **goldens** — DeepEval `Golden` objects stored as
JSON under `goldens/` — and each layer needs *two* kinds:

- **pass goldens**: `input` (+ `expected_output` and `context` where needed),
  no `actual_output`. The suite runs the *live* agent and expects a pass.
- **fail goldens**: same `input`, plus a canned `actual_output` and/or recorded
  `tools_called` simulating a specific failure mode. A mocked agent replays them
  and expects the layer to *catch* it. This is what turns the suite into a test
  of the eval harness itself — a judge that passes a planted contradiction is a
  broken judge, and the suite says so.

`additional_metadata` carries `question_id`, `section`, `scenario` (which
failure mode this golden models), and `expected_verdict` — the plumbing the
`expected_pass` metrics read.

Generation scripts:

| Script | Method | Output |
|---|---|---|
| `generate_goldens_from_context.py` | Synthesizer over `##` sections of the knowledge base; restricted evolutions (Reasoning/Concretizing/Comparative — Hypothetical and In-Breadth drift out of scope) + an LLM pass writing the `fail_contradiction`/`fail_incomplete` variants | `agent_judge_{pass,fail}_goldens.json` |
| `generate_fuzzy_goldens.py` | Derives key-phrase requirement lines per question | `agent_fuzzy_*.json` |
| `generate_structured_goldens.py` | `expected_tools` per question + recorded bad tool calls | `agent_structured_*.json` |
| `generate_goldens_from_scratch.py` | Synthesizer without context | `ai_engineering_goldens.json` |

`check_alignment` enforces the invariant the suite relies on: every pass query
appears exactly once per fail scenario, and `test_query_group_complete`
re-verifies it at test time.

---

## 6. Practical lessons (the part that makes this level-200)

These are the non-obvious findings baked into the code — each cost real
debugging time and each generalises beyond this repo.

### 6.1 Judge the parts, not the whole

Whole-answer correctness questions let fluent answers smuggle single false
claims past small decision models. Per-sentence `Choice`s + `min` aggregation
fixed a whole class of false passes. If your judge answers probabilistically,
give it the smallest decidable unit of work.

### 6.2 Keep per-request question counts small

More questions per call = fewer calls, but each question costs a model pass and
pushes latency toward the HTTP timeout on CPU. Four questions per call was the
empirical sweet spot here. Treat "questions per request" as a tunable next to
"max tokens" — and design your metrics so chunk boundaries are invisible.

### 6.3 Know your checkpoint's head quirks

Laya's `noul` head reads the literal `false:`/`true:` label token rather than
the state (tracked as laya issue #156) — verdicts come out wrong in a way that
looks like noise. The workaround is zero-cost: a two-option `Choice` over
`{"agrees": 1.0, "contradicts": 0.0}` computes the same P(statement holds)
through a code path that works. Lesson: when a model offers multiple output
heads, verify each against a known-answer probe before building on it.

### 6.4 Match the failure to the instrument

The hosted `jev-1.13` catches under-complete answers via its Score head. Laya's
`typed-decisions` Score head reads *topical overlap*, not coverage — a thin
on-topic answer still scores ~0.7. Rather than fight the model, the suite moved
the coverage gate to the fuzzy layer (where it was always going to be cheaper
anyway) and downgraded judge-side completeness for `fail_incomplete` to
report-only (`expected_pass=None`). Layered evals aren't just about cost —
different failure modes genuinely need different instruments.

### 6.5 Cold starts look like timeouts

A ~88s `ReadTimeout` on the first judge call after `laya-serve` starts is the
checkpoint loading, not a test failure. Poll `/health` for the `loaded` list in
CI before running the suite.

### 6.6 Small operational details

- DeepEval's results table crashes on `✓`/`⚠` glyphs through a cp1252
  console; `conftest.py` fixes it by reconfiguring stdout/stderr to UTF-8
  at import — no `PYTHONUTF8=1` needed (setting it anyway is harmless).
- `TYPESAFE_API_KEY` must be non-empty even though `laya-serve` doesn't check it
  (unless started with `LAYA_API_KEY` set) — the SDK validates client-side.
- `CONFIDENT_TRACE_VERBOSE=0` must be set *before* importing deepeval.
- Keep validator constants (`agent.MAX_SEARCH_RESULTS`) imported from the agent,
  not duplicated — the "shows N, limit is M" check reads the live constant.
- DeepEval **reconstructs metrics** passed to `assert_test` from their
  `__init__` signatures (`copy_metrics`) — every instance attribute whose
  name matches an `__init__` parameter in the MRO gets passed back in, so
  custom metrics must declare all such attributes as parameters (this is
  why the `*_util` metrics all take `threshold`).

### 6.7 Keep judge states small: big context bags hide contradictions

The grounding/faithfulness checks in `test_agent_extended.py` produced the
same lesson from a different angle. Pointing `JevEval` at the full
retrieval (a `CONTEXT` state of up to 24 tool-output paragraphs) both blew
the request timeout *and* blinded the checkpoint: a blatant contradiction
("PII refers to public infrastructure information") scored ~0.8 "agrees"
against a 7-paragraph context, and ~0.8 even when that same context was
moved into the `expected_output` state — the problem is state *size and
focus*, not which field carries it or how the question is phrased. Against
the single relevant paragraph it scored ~0.34 and was caught.

`GroundingMetric` therefore groups each answer's sentences by their
most-overlapping retrieved paragraph and judges every sentence against
*that paragraph alone*. Same decompose-and-verify rule as §6.1, applied to
the state rather than the answer: give the judge the smallest decidable
unit of evidence.

A related extractive-backend quirk, now fixed: the stem fallback used to
turn "bread" into `"brea"`, which prefix-matched "break"-family words,
retrieved paragraphs, and produced an answer instead of a refusal.
`stem_is_related` now verifies that words matched by a stem share the
query word's lexeme — same Porter stem or `fuzz.ratio` ≥ 85 ("bread"~"break"
scores exactly 80; real variants like "summarization"~"summarisation" ~96).
Residual caveat: a query word that appears *verbatim* in the knowledge
base in an unrelated sense still retrieves a paragraph, so out-of-scope
test queries should still be vetted for word-level absence.

---

## 7. Running it

```shell
uv sync
cp .env.example .env          # defaults already target local Laya + extractive agent

# terminal 1 — the judge (leave running)
.venv/Scripts/python run_judge.py
curl http://127.0.0.1:8000/health   # wait for "loaded": ["typed-decisions"]

# terminal 2 — smoke test, then the full suite
.venv/Scripts/deepeval.exe test run test_first_eval.py
.venv/Scripts/deepeval.exe test run test_agent.py           # 72 eval cases (80 pytest items)
.venv/Scripts/deepeval.exe test run test_agent_extended.py  # 50 evals + checks (54 evals with AGENT_TOOLS=search_knowledge,list_sections)

# subsets
.venv/Scripts/deepeval.exe test run test_agent.py -k "judge"
.venv/Scripts/deepeval.exe test run test_agent.py -k "q3-fuzzy-pass"
```

Layer toggles (`RUN_STRUCTURED_EVALS`, `RUN_FUZZY_EVALS`, `RUN_JEV_EVALS`) and
`SEQUENCE_FAIL_FAST` are module constants in `test_agent.py`. Switching the
judge backend to hosted Jev is a three-line `.env` change; nothing else.

---

## 8. Source material map — turning this into level-200 posts

Each section of this document maps naturally to a standalone intermediate-level
post. Suggested split, in dependency order:

1. **"Why agent evals need layers"** — §1, §4.4. The cost/verdict matrix and
   fail-fast sequencing; the core mental model.
2. **"Decision models vs. LLM-as-judge"** — §2, §3. Choice/Score/Noul, states,
   probabilities-instead-of-prose; `JevEval` under the hood.
3. **"One wire protocol, two judges: Jev and Laya"** — §2.2–2.4. Hosted ↔ local
   portability, `run_judge.py`, the `/v1/systemone` contract; good fit for
   AWS-adjacent "bring your own model endpoint" narratives.
4. **"Testing what the agent did, not just what it said"** — §4.1. Trace spans,
   per-span metrics, contract validation, `ToolCorrectnessMetric`.
5. **"Cheap checks first: key-phrase matching that isn't regex"** — §4.2. The
   matcher cascade and its cutoffs.
6. **"Teaching a judge to catch lies: decomposition, sharing, and
   expected-failure testing"** — §3.1–3.3, §4.3, §5. `SharedJevDecision`,
   per-sentence Choices, min-aggregation, negative goldens.
7. **"Model quirks in production eval pipelines"** — §6. Noul-head bug, score
   head vs. coverage, cold starts — the "lessons learned" war-story post.

## Appendix: file map

| File | Role |
|---|---|
| `agent.py` | ReAct agent under test; `llm` and deterministic `extractive` backends; `search_knowledge` tool + opt-in `list_sections` via `AGENT_TOOLS` |
| `agent-knowledge.md` | Markdown knowledge base; the agent's only data source |
| `run_judge.py` | Launches `laya-serve` on 127.0.0.1:8000 (Jev wire protocol) |
| `test_first_eval.py` | Minimal JevEval smoke test (hard-coded answer) |
| `test_first_eval_geval.py` | Same eval via GEval (any LLM judge — OpenAI, Ollama, …), for comparison; skips when none is configured |
| `test_agent.py` | The layered suite: stage definitions, sequencing, judge question construction |
| `test_agent_extended.py` | Behavioural extension: trajectory, keyword relevance, refusal, `GroundingMetric`, multi-turn, stability, tool selection |
| `deepeval_tool_util.py` | Span-level tool-call validation + trace plumbing (`tool_spans`, `attach_tool_metrics`, `record_tools_called`) |
| `conftest.py` | UTF-8 stdout/stderr reconfiguration for cp1252 Windows consoles |
| `deepeval_fuzzy_util.py` | `FuzzyMatchMetric`: exact → RapidFuzz → BLEU/ROUGE cascade |
| `deepeval_jev_util.py` | `SharedJevDecision` + `JevQuestionMetric` over `JevEval` outcomes |
| `generate_*.py` | Golden-set generators (context, scratch, fuzzy, structured) |
| `goldens/` | Committed golden JSON per layer, pass + fail sets |
| `how-to-test.md` | Operator runbook; the docstrings in the files above are the design rationale |
