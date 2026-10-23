# Agent Evals — What We're Testing and Why

This document explains the eval suite in this repo for someone joining the
team: what the system under test is, what each test file does, what data it
uses, what results to expect, and — most importantly — *why* each piece
exists. Read it top to bottom once; afterwards use it as a reference.

For the pure runbook (commands, flags, troubleshooting), see
[`how-to-test.md`](how-to-test.md). This file is about understanding.

---

## 1. The 30-second version

We have a small AI agent (`agent.py`) that answers beginner questions about
AI engineering by searching a Markdown knowledge base
(`agent-knowledge.md`). Like any LLM system, it can fail in ways that plain
unit tests can't catch: it can answer fluently but *wrongly*, answer
*vaguely*, call its tool with bad arguments, or skip searching entirely.

This repo is a suite of **evaluations ("evals")** — tests that measure the
*quality* of the agent's behaviour, not just whether the code runs. It
checks the agent at three levels, cheapest first:

1. **Structured** — did the agent call its search tool correctly? (pure rules, free)
2. **Fuzzy** — does the answer cover the required key points? (local NLP, free)
3. **Judge** — does the answer agree with a reference answer, sentence by
   sentence? (a decision model, one call per test)

The suite also **tests itself**: alongside live-agent checks, every layer
is fed *deliberately wrong* canned answers that it must catch. If the judge
passes a planted contradiction, the eval harness — not the agent — is
broken, and the suite fails. This is the key idea that makes the results
trustworthy.

---

## 2. The system under test: `agent.py`

`agent.py` is a minimal **ReAct agent** — no framework, just the OpenAI SDK
plus a loop:

```
user query ──► LLM produces a JSON "step": {thought, action, keywords, answer}
                  │
                  ├─ action = "search_knowledge" ──► run tool, feed results back ──► next step
                  │    ("list_sections" too, when enabled via AGENT_TOOLS)
                  │
                  └─ action = "final_answer"      ──► return answer
```

- **Tools**: `search_knowledge(keywords)` — keyword search over the
  paragraphs of `agent-knowledge.md` (our "knowledge base", a curated doc on
  AI engineering topics: PII, temperature/top-p, memory, tokens, RAG,
  observability, etc.). Tool contract: 1–5 keywords, each a single word.
  An optional second tool, `list_sections()`, returns the knowledge base's
  section outline for meta-questions ("What topics do you cover?"); it is
  enabled via `AGENT_TOOLS=search_knowledge,list_sections` and is off by
  default (see §5.5).
- **Guardrails**: `MAX_STEPS = 5` ReAct steps per query,
  `MAX_SEARCH_RESULTS = 24` paragraphs per search, last 5 user turns of
  context.
- **Two backends** (chosen by `AGENT_BACKEND` in `.env`):
  - `llm` — a real model at any OpenAI-compatible endpoint (OpenAI, Ollama,
    LM Studio, vLLM…).
  - `extractive` — a deterministic stand-in that makes the same tool calls
    and stitches an answer from the retrieved paragraphs. **Zero API keys
    needed.** This is the default, and it's what makes most of the suite
    runnable fully offline.
- **Tracing**: the loop, LLM calls, and tool calls are wrapped in
  DeepEval's `@observe` decorators, producing a *trace* of spans per run.
  The structured eval layer inspects these spans — this is how we test what
  happened *inside* a run, not just the final text.

### The failure modes we care about

The suite is built around named scenarios, each encoded in golden
metadata:

| Scenario | What it simulates | Which layer must catch it |
|---|---|---|
| `pass` | the real agent doing its job | every layer must pass it |
| `fail_invalid_keywords` | tool called with malformed keywords (multi-word, or >5) | structured |
| `fail_irrelevant_results` | search returned paragraphs containing none of the keywords | structured |
| `fail_no_tool_call` | agent answered without ever searching | structured (via ToolCorrectnessMetric) |
| `fail_contradiction` | fluent answer containing a false claim | judge |
| `fail_incomplete` | truthful but thin/vague answer | fuzzy (gated); judge reports only |

Why both "pass" and "fail" goldens? A test that only ever sees good answers
proves nothing — a broken metric that always returns 1.0 would look
perfect. Deliberately failing inputs prove the metric can actually say "no".

---

## 3. The judge: JevEval / Laya (and how it differs from GEval)

The third layer needs something that can judge *meaning*, which rules and
keyword matching can't do. Two judge technologies appear in this repo:

| | JevEval (decision model) | GEval (LLM-as-judge) |
|---|---|---|
| What it is | A small classifier-style model that answers structured questions about a test case and returns **probabilities** for each option | A general LLM prompted to grade the answer, producing prose that gets parsed into a score |
| Used by | `test_first_eval.py`, `test_agent.py`, `test_goldens_from_scratch.py` | `test_first_eval_geval.py` |
| Backend here | **Laya** served locally by `run_judge.py` on `http://127.0.0.1:8000`, speaking DeepEval's `/v1/systemone` wire protocol (or hosted Jev via `TYPESAFE_*` env vars) | OpenAI by default (`OPENAI_API_KEY`), or any local model via `deepeval set-ollama` |
| Cost | ~0.2–0.5s per decision on CPU, free | an API call (or a loaded local model) per metric |

### JevEval's question types

A `JevEval` metric is configured with `evaluation_params` (which fields of
the test case to look at — here `ACTUAL_OUTPUT` vs `EXPECTED_OUTPUT`) and a
list of **questions**. Three question forms appear in this codebase:

- **`Choice`** — "Does this statement agree with expected_output?" with
  options `{"agrees": 1.0, "contradicts": 0.0}`. The judge returns a
  probability per option; the score is the probability-weighted value.
- **`Score`** — "How much of the detail in expected_output is present in
  actual_output?" with ordered levels
  `["Almost none", "Some", "Most", "All of it"]` mapped to 0→1.
- **`Noul`** — a yes/no statement ("No fact in actual_output contradicts
  a fact in expected_output"). ⚠️ **Broken on the local `typed-decisions`
  checkpoint** (laya issue #156 — it reads the label token instead of the
  state). That's why the main suite uses two-option `Choice`s instead;
  `test_goldens_from_scratch.py` still uses `Noul`, which is why it fails
  under the local judge.

### Design subtleties you'll see in the code

- **Contradiction is checked per sentence, not per answer.** A fluent
  paragraph with one blatant falsehood can score ~0.7 "agrees" on a
  whole-answer judgement. Splitting the answer into sentences and taking
  the *minimum* verdict (decompose-and-verify) means one bad sentence sinks
  the whole answer.
- **Questions are chunked at 4 per judge call** (`QUESTIONS_PER_CALL`):
  every question costs the model a pass over the input states, and one
  giant call can exceed the request timeout on a CPU-resident checkpoint.
- **`SharedJevDecision` / `JevQuestionMetric`** (`deepeval_jev_util.py`):
  DeepEval would normally make one judge call per metric. These helpers run
  JevEval *once per test case* (per chunk) and then split the per-question
  outcomes out as separate named metrics ("No contradiction",
  "Completeness") — same information, a fraction of the calls.
- **`expected_pass` semantics**: every custom metric here can be told what
  verdict is *expected*. `expected_pass=True/False` means "the test passes
  iff the metric's verdict matches". `expected_pass=None` means "score it
  and report it, but don't let it decide pass/fail" — used for completeness
  on `fail_incomplete` goldens, where the local checkpoint can't grade
  coverage reliably.

---

## 4. The data: "goldens"

**Goldens** are DeepEval's name for test data records. In this repo they're
JSON files under `goldens/`, loaded via `EvaluationDataset`. A golden
carries an `input` (the user query) plus whichever of
`actual_output` / `expected_output` / `tools_called` / `expected_tools` /
`context` the layer needs, and `additional_metadata` with:

- `question_id` — the query it belongs to (`q1`–`q8` in the layered suite;
  the extended files add `q9`–`q11` and `q20`–`q23`)
- `scenario` — `pass`, `fail_contradiction`, `fail_incomplete`,
  `fail_invalid_keywords`, `fail_irrelevant_results`, `fail_no_tool_call`;
  the extended goldens add `refuse_out_of_scope`, `fail_answered_anyway`,
  and `kind`-tagged records (`multipart`, `tool_selection`)
- `expected_verdict` — what the eval *should* conclude
- `section` — the knowledge-base section the query was generated from

| File | Records | Feeds | Contents |
|---|---|---|---|
| `agent_structured_pass_goldens.json` | 8 (1/query) | structured layer | `expected_tools` = `[search_knowledge]`; the live agent supplies real calls |
| `agent_structured_fail_goldens.json` | 24 (3/query) | structured layer | `tools_called` = canned bad tool calls the mocked agent *replays* into the trace |
| `agent_fuzzy_pass_goldens.json` | 8 (1/query) | fuzzy layer | `expected_output` = key-phrase requirements (one per line, `a \| b` = alternatives) |
| `agent_fuzzy_fail_goldens.json` | 8 (1/query) | fuzzy layer | `actual_output` = thin-but-truthful canned answers; same key-phrase format |
| `agent_judge_pass_goldens.json` | 8 (1/query) | judge layer | `expected_output` = full reference answer; `actual_output` left empty — the live agent answers |
| `agent_judge_fail_goldens.json` | 16 (2/query) | judge layer | `actual_output` = canned `fail_contradiction` and `fail_incomplete` answers |
| `agent_extended_goldens.json` | 4 | `test_agent_extended.py` | 2 out-of-scope refusal queries (q9, q10), 1 canned "answered anyway" negative, 1 multi-part in-scope query (q11) |
| `agent_tool_selection_goldens.json` | 4 | `test_agent_extended.py` | `expected_tools` per query — `list_sections` for 3 coverage questions, `search_knowledge` for 1 topical control (q20–q23) |
| `ai_engineering_goldens.json` | 9 (3 questions × 3 scenarios) | `test_goldens_from_scratch.py` | self-contained: input + actual + expected in every record |

**Where they came from** (regenerate only if needed — requires an LLM,
usually `OPENAI_API_KEY`):

- `generate_goldens_from_scratch.py` → `ai_engineering_goldens.json`:
  DeepEval's Synthesizer invents questions from nothing, then a prompt
  writes a reference answer plus three variants (passing / contradictory /
  incomplete).
- `generate_goldens_from_context.py` → `agent_judge_{pass,fail}_goldens.json`:
  questions + reference answers grounded in sections of
  `agent-knowledge.md`.
- `generate_structured_goldens.py` → `agent_structured_*.json`: builds the
  canned good/bad tool calls deterministically (no LLM).
- `generate_fuzzy_goldens.py` → `agent_fuzzy_*.json`: an LLM extracts the
  key-phrase requirements from each reference answer.

---

## 5. The test files, in depth

### 5.1 `test_first_eval.py` — the smoke test (1 test)

```
deepeval test run test_first_eval.py   →  1 passed in a few seconds
```

**What it does:** builds one `LLMTestCase` — question "What is a context
window?", a hard-coded `actual_output` (correct but brief) and a fuller
`expected_output` — and runs one `JevEval` metric over it:

- `Choice`: "Does every fact in actual_output agree with expected_output?"
  (weight 2 — agreement counts double)
- `Score`: "How much of the detail in expected_output is present in
  actual_output?" (4 levels)

Threshold 0.5; the weighted mean must reach it. With the local judge this
typically scores ~0.67 and **passes**.

**Why it exists:** it's the cheapest possible end-to-end proof that the
judge pipeline works — `.env` loaded, `run_judge.py` serving, the TypeSafe
SDK talking to it, DeepEval reporting. If the eval stack is broken, this
one test tells you in seconds before you run the 3-minute suite. Run it
first every time you change environment.

### 5.2 `test_agent.py` — the layered suite (80 tests)

```
deepeval test run test_agent.py        →  72 eval cases + 8 data checks, ~3–4 min
```

This is the heart of the repo. For each of the 8 queries it runs the layers
**in sequence order — cheapest first, judge last** (fail fast: once a layer
fails for a query, that query's later layers are skipped via
`SEQUENCE_FAIL_FAST`).

#### Layer 1 — structured (`deepeval_tool_util.py`), 32 tests

Judges the *tool calls inside the run*, not the answer. DeepEval's pytest
plugin gives each test a trace; `attach_tool_metrics` attaches a
`ToolCallValidationMetric` to every `search_knowledge` tool span, and
`record_tools_called` copies the spans into `tools_called` so
`ToolCorrectnessMetric` can compare them against the golden's
`expected_tools`.

`ToolCallValidationMetric` checks per call (pure rules, deterministic):

- **input**: valid JSON `{"keywords": [...]}`, 1–5 keywords, each a single
  non-empty word (mirrors the tool contract enforced by `STEP_SCHEMA`)
- **output**: either a "no match" message, or the summary line
  `N paragraphs matched; showing M.` followed by exactly M result blocks
  (M ≤ N, M ≤ `MAX_SEARCH_RESULTS`), each block = heading lines + paragraph
- **relevance**: every returned paragraph contains at least one keyword

Tests per query:

| Test id | What happens | Expected outcome |
|---|---|---|
| `qN-structured-pass` | live agent runs fresh inside the test's trace | calls valid + `ToolCorrectnessMetric` passes |
| `qN-structured-fail_invalid_keywords` | mocked agent replays calls with `["privacy and data handling", ...]` — a multi-word keyword and >5 keywords | validation **must fail**; tool correctness still passes (a call happened) |
| `qN-structured-fail_irrelevant_results` | replayed call returns paragraphs containing none of the keywords | validation **must fail** |
| `qN-structured-fail_no_tool_call` | mocked agent records zero tool spans | `ToolCorrectnessMetric` **must fail** (per-call validation can't see an absent call) |

The mocked "agent" (`replay_tool_calls`, `@observe(type="agent")`) replays
the golden's recorded `tools_called` as real tool spans, so the trace looks
exactly like a live run's. `ExpectedVerdictMetric` wraps DeepEval metrics
so "metric correctly failed" counts as a test pass.

#### Layer 2 — fuzzy (`deepeval_fuzzy_util.py`), 16 tests

Key-phrase coverage of the answer. The golden's `expected_output` isn't a
reference text — it's **requirements**: one per line, with `|` separating
alternative phrasings of the same requirement
(`"PII | personally identifiable information"`). A requirement is met if
*any* alternative is found in the answer, via a cascade of matchers,
cheapest first:

1. **exact** — normalized phrase appears as whole words
2. **rapidfuzz** — best similarity ratio over same-length word windows ≥
   85 (catches typos/inflections; skipped for phrases < 5 chars, where one
   letter is too big a share — "rag" vs "drag" = 86)
3. **n-gram** — BLEU or ROUGE over Porter-stemmed tokens ≥ 0.6 (catches
   reordering and different word forms)

`FuzzyMatchMetric.score` = fraction of requirements met; threshold 0.6.

| Test id | Expected outcome |
|---|---|
| `qN-fuzzy-pass` | live answer covers ≥60% of the required phrases → pass |
| `qN-fuzzy-fail_incomplete` | canned thin answer covers almost none → metric must fail |

**Why this layer exists:** fuzzy matching checks *which concepts the answer
covers*, not whether statements are true — so it owns `fail_incomplete`
gating (the local judge's score head can't grade coverage) while leaving
contradictions to the judge. It's also ~free, so it runs before the judge.

#### Layer 3 — judge (`JevEval` via `deepeval_jev_util.py`), 24 tests

For each answer:

1. Split `actual_output` into sentences.
2. Build one `Choice("Does this statement agree with expected_output: \"<sentence>\"")`
   per sentence + one shared `Score` completeness question.
3. Send questions to Jev in chunks of 4, each chunk wrapped in a
   `SharedJevDecision` (one call, shared outcomes).
4. Report two metrics per test:
   - **"No contradiction"** = min over all sentence verdicts
   - **"Completeness"** = the score question's value

| Test id | Expected outcome |
|---|---|
| `qN-judge-pass` | live answer: no contradiction AND completeness ≥ 0.5 → pass |
| `qN-judge-fail_contradiction` | canned fluent answer with one false claim → "No contradiction" must fail |
| `qN-judge-fail_incomplete` | canned thin answer → "No contradiction" must pass (it's truthful); completeness is **report-only** (`expected_pass=None`) under the local checkpoint |

#### Data integrity — `test_query_group_complete[qN]` (8 tests)

Plain pytest asserts (not in DeepEval's results table): each query must
have exactly one pass golden plus one golden per failure mode each enabled
layer owns. Catches a malformed/edited golden file before it silently
weakens the suite.

#### Wiring details worth knowing

- Answers are **cached per query** (`_answers`): the structured layer runs
  the live agent `fresh=True` inside its trace, and fuzzy/judge reuse that
  same answer — one agent run per query, all layers judging the same run.
- Test IDs are `qN-<layer>-<scenario>`, which is what makes `-k` filtering
  so useful: `-k "q3"`, `-k "structured and fail"`,
  `-k "q3-fuzzy-pass"`…
- Layer toggles at the top of the file: `RUN_STRUCTURED_EVALS`,
  `RUN_FUZZY_EVALS`, `RUN_JEV_EVALS`, `SEQUENCE_FAIL_FAST`.
- **Don't use `-n` (pytest-xdist)**: fail-fast sequencing keeps per-query
  state in memory; xdist scatters tests across processes.

### 5.3 `test_goldens_from_scratch.py` — the older flat suite (9 tests)

```
deepeval test run test_goldens_from_scratch.py   →  ⚠ expect failures under local Laya
```

The tutorial-04 artifact: runs `ai_engineering_goldens.json` (3 questions ×
pass/fail_contradiction/fail_incomplete) through one judge call per test
reporting two metrics: "No contradiction" (a `Noul`) and "Completeness"
(a `Score`).

**Expected under the local `typed-decisions` checkpoint: failures.** This
is a *checkpoint limitation, not a harness bug*:

- `Noul` is broken on laya (issue #156) — the main suite works around it
  with `Choice`s; this file predates the fix.
- `fail_incomplete` expects the `Score` head to grade coverage; laya reads
  topical overlap instead and rates thin on-topic answers ~0.7 → fail.

Against hosted `jev-1.13` (three-line `.env` change to `TYPESAFE_*`) it
should pass 9/9. Its value in this repo is demonstrating both that the
`Noul` question form exists and exactly where the local checkpoint's edges
are.

### 5.4 `test_first_eval_geval.py` — the comparison test (1 test)

```
deepeval test run test_first_eval_geval.py   →  needs an LLM judge
```

Identical test case to `test_first_eval.py` (context window question, same
actual/expected outputs), but judged by **`GEval`** — a generative
LLM-as-judge that prompts a model to grade the answer, instead of a
decision model returning probabilities over states.

**Why it exists:** the contrast case. It lets you compare, on the same
input, the two judging philosophies — `GEval`'s prose+parse approach vs.
Jev's structured questions+probabilities. Runs when `OPENAI_API_KEY` is a
real key, or a local model is configured (`deepeval set-ollama`,
`LOCAL_MODEL_*`/`OLLAMA_*` env vars, or an explicit `model=` in code);
**skips cleanly when none is** (`_judge_model_configured` checks env vars
and deepeval's persisted provider config). It does *not* use
`run_judge.py`.

### 5.5 `test_agent_extended.py` — the behavioural extension (62 tests)

```
deepeval test run test_agent_extended.py   →  ~3.5 min; judge needed for grounding only
```

The main suite checks tool-call *form* and answer *quality*; this file
probes how the agent *behaves*. It adds five small custom metrics (same
`expected_pass` plumbing as the `*_util` metrics) plus plain pytest checks
where a metric adds nothing. Data: `goldens/agent_extended_goldens.json`
and `goldens/agent_tool_selection_goldens.json`, plus the existing judge
pass/fail sets.

| Test | Count | What it proves |
|---|---|---|
| `test_trajectory_and_keywords[qN-pass]` | 9 (q1–q8 + q11) | Live run: ≥1 tool call (searched before answering — the degenerate "tool selection" check for a one-tool agent), ≤4 calls (bounded), converged (no `MAX_STEPS` fallback), and ≥50% of searched keyword words grounded in the query or context — a word counts if it appears inside a query word (covers stems like "summar" for "summarization") or anywhere in the context (`TrajectoryMetric` + `KeywordRelevanceMetric`) |
| `test_trajectory_flail_caught` | 1 | Canned agent with 5 tool calls → budget check **must fail** |
| `test_lazy_refusal_caught` | 1 | Refusal text without any search → refusal metric passes but "searched first" **must fail** — right words, wrong behaviour |
| `test_keyword_relevance_irrelevant_caught` | 1 | Well-formed keywords "zebra"/"quantum" → relevance **must fail** |
| `test_out_of_scope_refusal[q9/q10]` | 2 | Out-of-scope queries (capital of France, Mariana Trench) → the refusal string, and only *after* searching the KB — with a tighter budget of ≤3 calls since a refusal needs no second search (`RefusalMetric` + `TrajectoryMetric`). The metric accepts the exact string or the looser `"cannot help with that"` fragment, so the `llm` backend's phrasing variance doesn't flake it |
| `test_out_of_scope_answered_anyway_caught[q9]` | 1 | Canned confident answer to an out-of-scope query → refusal check **must fail** |
| `test_grounding[qN-pass]` | 9 | Faithfulness: every sentence of the live answer agrees with the retrieved paragraphs, judged per-sentence vs. its best-matching paragraph, `min` over the sentence verdicts (`GroundingMetric`) |
| `test_grounding_catches_unsupported` | 16 | Canned answers vs. their context: `fail_contradiction` must fail grounding; `fail_incomplete` is still grounded (grounding ≠ completeness) |
| `test_answer_stability[qN-pass]` | 9 | 3 fresh runs per query: identical under `extractive`, ≥85% pairwise RapidFuzz under a real model; also fails if any run hits the `MAX_STEPS` fallback or refuses an in-scope question (`StabilityMetric`) |
| `test_multi_turn_history_accumulates` / `test_multi_turn_window_capped` | 2 | Plain pytest: history keeps both turns; capped at `MAX_USER_TURNS` after 7 queries |
| `test_multi_turn_followup` / `test_llm_backend_smoke` | 2 | `skipif` unless `AGENT_BACKEND=llm`: pronoun follow-up resolution, and a real-model tool-choice smoke test |
| `test_extractive_tool_choice[qN]` | 4 | Plain pytest (no trace needed): the extractive policy's first step picks the expected tool — `list_sections` for coverage questions, `search_knowledge` for topical ones. With the second tool disabled it asserts the flag-off fallback instead |
| `test_tool_selection[qN]` | 4 | `skipif` unless `AGENT_TOOLS` enables `list_sections`. Live run: `ToolCorrectnessMetric` matches the golden's `expected_tools` — real multi-tool selection, verified end to end |
| `test_tool_selection_wrong_tool_caught` | 1 | Canned agent answering a coverage question with keyword search → tool correctness **must fail** — wrong tool, even though it "tried" |

**Two lessons baked into the grounding check** (both measured, not
guessed):

- `JevEval`'s `CONTEXT` state with full tool output (up to 24 paragraphs)
  times out, and even at 7 paragraphs the local checkpoint scores a blatant
  contradiction ~0.8 "agrees" — it misses contradictions against large
  grab-bag states. `GroundingMetric` therefore groups answer sentences by
  their most-overlapping retrieved paragraph and judges each against that
  *single* paragraph (≤4 questions/call): same decompose-and-verify rule,
  applied to context. A contradiction then scores ~0.34 → caught.
- The extractive backend's stem fallback once turned "bread" into `"brea"`,
  which matched "break"-family words and produced an answer instead of a
  refusal. Stems are now verified (`stem_is_related` in `agent.py`): a stem
  is only used if the words it matched share the query word's lexeme —
  same Porter stem, or a RapidFuzz ratio ≥ 85 ("bread"~"break" scores 80,
  real spelling variants like "summarization"~"summarisation" score ~96).
  Residual risk: a query word that appears *verbatim* in the knowledge base
  in an unrelated sense still retrieves paragraphs — genuine out-of-scope
  queries should use words absent from `agent-knowledge.md`.

**Multi-tool selection** is covered when the opt-in second tool is enabled:
`AGENT_TOOLS=search_knowledge,list_sections` gives the agent a real choice
between `search_knowledge` and `list_sections`, and the selection tests
above verify the routing both ways. With the default single tool, the only
"selection" is search-before-answering, which the trajectory check asserts.
Not covered: real-LLM decision-making (the `llm`-gated tests cover it only
when configured).

**Traces under plain `pytest`:** the trace-reading tests call `ensure_trace()`,
which uses the per-test trace `deepeval test run` provides, or starts a fresh
`deepeval.tracing.trace()` when there is none — so they run under both
runners. (A manually started trace just isn't posted to Confident AI.)

---

## 6. What each command does and what to expect

```bat
REM window 1 — start the local judge, leave running (skip for offline runs)
.venv\Scripts\python run_judge.py

REM window 2 — smoke test, then the suite
.venv\Scripts\deepeval.exe test run test_first_eval.py            REM 1 test,  judge needed
.venv\Scripts\deepeval.exe test run test_agent.py                 REM 80 tests, judge for judge layer only
.venv\Scripts\deepeval.exe test run test_goldens_from_scratch.py  REM 9 tests,  judge needed, fails under local Laya (known limitation)
.venv\Scripts\deepeval.exe test run test_first_eval_geval.py      REM 1 test,   skips unless an LLM judge is configured
.venv\Scripts\deepeval.exe test run test_agent_extended.py        REM 62 tests, judge for grounding only

REM everything at once (--ignore=gitsss skips a clone of the project whose
REM duplicate test_*.py basenames break pytest collection)
.venv\Scripts\deepeval.exe test run . --ignore=gitsss

REM offline subsets — no judge, no keys, seconds
.venv\Scripts\deepeval.exe test run test_agent.py -k "not judge"   REM 56 tests

REM narrower slices
.venv\Scripts\deepeval.exe test run test_agent.py -k "structured"        REM 32 — tool-call layer
.venv\Scripts\deepeval.exe test run test_agent.py -k "judge"             REM 24 — judge layer only
.venv\Scripts\deepeval.exe test run test_agent.py -k "q3-fuzzy-pass"     REM 1 exact case
.venv\Scripts\deepeval.exe test run test_agent.py -k "fail_incomplete"   REM 16 — thin-answer gating
```

Each run writes `deepeval_results\test_run_<timestamp>.json`
(`DEEPEVAL_RESULTS_FOLDER`) — every test case with its metric scores,
reasons, and verdicts, which is where you go to dig into *why* something
failed (each metric reports e.g. `"min of 4 question(s) -> fail ...,
Probabilities: ..."` or `"3/5 key phrase requirements met ..."`).

---

## 7. Why we do this — the benefits

1. **LLM outputs can't be unit-tested.** `assert output == expected` fails
   on any rephrasing; evals grade *meaning* — agreement with a reference,
   coverage of required points, validity of behaviour — the way a human
   reviewer would, but automatically and repeatably.

2. **Regression safety for prompts, models and code.** Change the system
   prompt, swap the model, or refactor `agent.py` and rerun: the suite
   tells you which of 8 queries degraded, in which layer, and how. Without
   it you're shipping on vibes.

3. **Testing the process, not just the output.** The structured layer
   inspects tool calls through the trace, so "the answer looked fine but
   the agent searched badly / not at all" is caught — answer-only checks
   can't see that.

4. **The suite verifies itself.** Every layer is fed planted failures it
   must catch. This is the single most important design idea here: an eval
   harness that can't fail is worthless, and this one proves its own
   sensitivity on every run.

5. **Layering keeps it cheap and debuggable.** Free rule checks run first,
   local NLP second, model calls last, and fail-fast skips downstream
   layers for a broken query. You get fast feedback for CI (the
   `-k "not judge"` subset runs in seconds offline) and deep judgement when
   it matters.

6. **An honest map of the judge's limits.** The suite documents and encodes
   real, measured weaknesses of the local checkpoint (noul head, coverage
   grading) — with workarounds — rather than discovering them in
   production. The same code runs against hosted `jev-1.13` with a
   three-line `.env` change when stronger judging is needed.

---

## 8. Known limitations (so you're not surprised)

These are the checkpoint's own weaknesses — they can't be fixed locally,
only worked around (the suite does) or avoided (hosted `jev-1.13`):

- `test_goldens_from_scratch.py` **fails under the local Laya checkpoint**
  — the `Noul` question form reads the `false:`/`true:` label token instead
  of the state (laya#156, still open in laya 0.4.13), and its `Score` head
  can't grade coverage. Passes on hosted `jev-1.13`.
- The judge measures **topical overlap more than coverage**, so
  `fail_incomplete` in the judge layer reports completeness without gating
  on it — the fuzzy layer owns incomplete-answer gating.
- The judge also **misses contradictions against large context states** —
  which is why `GroundingMetric` checks each sentence against a single
  best-matching paragraph instead of the whole retrieval.
- First judge request after starting `run_judge.py` can take a minute or
  two (checkpoint loads lazily; `run_judge.py` sets `LAYA_PRELOAD=0` —
  flip it to 1 to pay the cost at startup instead). Check
  `curl http://127.0.0.1:8000/health`.

What used to be on this list and is now fixed:

- ~~Extractive stem false-matches~~ — stems are verified (`stem_is_related`):
  "bread"→"brea"→"break" no longer produces an answer. Residual: a query
  word that appears *verbatim* in the KB in an unrelated sense still matches.
- ~~Multi-tool selection untested~~ — `AGENT_TOOLS=search_knowledge,list_sections`
  enables a second tool and the extended file's selection tests.
- ~~Trace tests skip under plain `pytest`~~ — `ensure_trace()` starts a
  fresh trace when none exists. The *structured* layer in `test_agent.py`
  still needs `deepeval test run` (its span metrics are evaluated by the
  test-run plugin) and now **skips cleanly** there instead of erroring.
- ~~`PYTHONUTF8=1` required~~ — `conftest.py` reconfigures stdout/stderr to
  UTF-8 at import; setting it anyway is harmless.
- ~~`test_first_eval_geval.py` errors without a key~~ — it **skips** unless
  an LLM judge is configured (real `OPENAI_API_KEY`, `LOCAL_MODEL_*`/
  `OLLAMA_*` env vars, or a provider saved by `deepeval set-*`). A *real
  but invalid* key still fails at request time — the skip can't detect
  that without calling out.

---

## 9. File map

| File | Role |
|---|---|
| `agent.py` | System under test: ReAct loop, `search_knowledge` + opt-in `list_sections` tools (`AGENT_TOOLS`), `llm`/`extractive` backends, `@observe` tracing |
| `agent-knowledge.md` | The knowledge base the agent searches (source of truth for its answers) |
| `run_judge.py` | Starts the local judge (Laya served over the Jev `/v1/systemone` protocol) |
| `test_first_eval.py` | 1-test smoke test of the judge pipeline |
| `test_agent.py` | The 80-test layered suite (structured → fuzzy → judge, fail-fast) |
| `test_goldens_from_scratch.py` | 9-test flat judge suite; known-failing under local Laya |
| `test_first_eval_geval.py` | 1-test GEval comparison of judge styles |
| `test_agent_extended.py` | 62-test behavioural extension: trajectory, keyword quality, refusal, grounding, multi-turn, variance, multi-part, tool selection (opt-in) |
| `deepeval_jev_util.py` | `SharedJevDecision` (one judge call shared across metrics), `JevQuestionMetric` (per-question reporting) |
| `deepeval_fuzzy_util.py` | `FuzzyMatchMetric` + the exact → RapidFuzz → stemmed BLEU/ROUGE matching cascade |
| `deepeval_tool_util.py` | `ToolCallValidationMetric`, `ExpectedVerdictMetric`, trace helpers (`attach_tool_metrics`, `record_tools_called`, `tool_spans`) |
| `conftest.py` | Reconfigures stdout/stderr to UTF-8 so the results table prints its Unicode marks on cp1252 consoles without `PYTHONUTF8=1` |
| `goldens/*.json` | The test data (see §4), incl. `agent_tool_selection_goldens.json` for the opt-in second tool |
| `generate_*.py` | Golden-set generators (need an LLM; outputs are committed) |
| `.env` / `.env.example` | Judge backend (`TYPESAFE_*`), agent backend (`AGENT_BACKEND`, `AGENT_TOOLS`, `INFERENCE_*`), `OPENAI_API_KEY`, `DEEPEVAL_RESULTS_FOLDER` |
| `deepeval_results/` | Per-run JSON results with per-metric scores and reasons |
| `how-to-test.md` | The runbook: commands, flags, troubleshooting |
| `jev-eval-technical-guide.md` | Deeper reference on the Jev/judge mechanics |
