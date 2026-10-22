# How to run the tests

This repo evaluates `agent.py` — a small ReAct agent that answers beginner
AI-engineering questions by searching `agent-knowledge.md` — in three layers:

| Layer | Cost | What it checks | Where |
|---|---|---|---|
| 1. structured | ~0 (rules) | the `search_knowledge` tool calls are well-formed and actually happened | `deepeval_tool_util.py` |
| 2. fuzzy | ~0 (local NLP) | the answer covers the required key phrases | `deepeval_fuzzy_util.py` |
| 3. judge | one decision-model call per test | the answer agrees with a reference answer, sentence by sentence | `JevEval` → local Laya or hosted Jev |

The suite is also a test of the eval harness itself: besides running the
**live** agent on "pass" goldens, every layer gets "fail" goldens that a
mocked agent replays — planted failure modes the layer is *expected to
catch*. A judge that passes a planted contradiction means the harness is
broken, and the suite fails accordingly.

With the default `.env.example` everything runs locally and offline — the
agent uses the deterministic `extractive` backend (no LLM) and the judge is
a local Laya server — so no API keys are needed for the main suite.

## The test files

There are **five** test files at the repo root — the technical guide's
`generate_*`/`deepeval_*` files are utilities and data generators, not
tests. 144 pytest tests in total:

| File | Tests | Judge needed? | API key? | What it does |
|---|---|---|---|---|
| `test_first_eval.py` | 1 | yes | no | Smoke test: one hard-coded answer vs. a reference, through `JevEval`. Proves the judge connection works end to end. |
| `test_agent.py` | 80 (72 evals + 8 data checks) | only for the judge layer | no (with `extractive` backend) | The layered suite: structured → fuzzy → judge, over queries q1–q8. |
| `test_agent_extended.py` | 53 (49 evals + 2 pytest checks + 2 llm-gated) | only for grounding | no (with `extractive` backend) | Behavioural extension: trajectory/step budget, keyword quality, refusal path, grounding vs. retrieval, multi-turn memory, variance, one multi-part query. |
| `test_goldens_from_scratch.py` | 9 | yes | no | Judge metrics over `goldens/ai_engineering_goldens.json`. Uses `Noul`, so it needs the **hosted Jev** to pass fully — see its section below. |
| `test_first_eval_geval.py` | 1 | no | an LLM judge (`OPENAI_API_KEY` or Ollama/local — see its section) | The same correctness eval via `GEval` (generative LLM-as-judge), for comparison with the decision-model judge. |

## Which environment each test needs

The judge backend and agent backend are chosen by `.env` alone — no code
changes:

| Configuration | `.env` settings | What it enables |
|---|---|---|
| **Local Laya + extractive agent** (default, fully offline) | `TYPESAFE_BASE_URL=http://localhost:8000`, `TYPESAFE_API_KEY=local-laya`, `TYPESAFE_MODEL_NAME=typed-decisions`, `AGENT_BACKEND=extractive` | Everything except `test_first_eval_geval.py`. The only setup this runbook assumes. |
| **Hosted Jev** | `TYPESAFE_BASE_URL=<gateway>`, `TYPESAFE_API_KEY=<key>`, `TYPESAFE_MODEL_NAME=jev-1.13` | Same tests, no `run_judge.py` needed. Additionally `test_goldens_from_scratch.py` passes fully: jev-1.13's `Noul` head works and its `Score` head catches incomplete answers — two things the local checkpoint can't do (see "test_goldens_from_scratch.py" below). |
| **Real LLM agent** | `AGENT_BACKEND=llm` + `INFERENCE_*` (defaults point at Ollama: `http://localhost:11434/v1`, `llama3.2`; `INFERENCE_API_STYLE=chat` also covers LM Studio/vLLM, `responses` covers OpenAI) | The "pass" tests in `test_agent.py` now exercise a real model instead of the deterministic extractive stand-in, so pass/fail depends on that model's quality. Planted-failure tests are unaffected (they use mocked answers). |
| **LLM-as-judge for GEval** | `OPENAI_API_KEY=<key>` **or** a local model — see "test_first_eval_geval.py" below | `test_first_eval_geval.py`, and regenerating goldens with `generate_*.py` (DeepEval's Synthesizer defaults to OpenAI). |

Notes:

- You can mix these: e.g. local Laya + `AGENT_BACKEND=llm` exercises a real
  agent while keeping the judge free and local.
- `TYPESAFE_API_KEY` must be non-empty even for local Laya — the TypeSafe
  SDK validates it client-side; `laya-serve` itself ignores it unless
  started with `LAYA_API_KEY` set.
- Want to eyeball what the agent does? `.venv/Scripts/python agent.py`
  runs it interactively.

## Prerequisites

1. The virtualenv exists (`.venv/`) with `deepeval` and `laya[serve]` installed.
2. `.env` exists — `copy .env.example .env` if not. The defaults already
   point at the local judge and the extractive agent.
3. Set `PYTHONUTF8=1` in every window you run tests in — otherwise
   DeepEval's results table crashes printing `✓`/`⚠` through the cp1252
   console. (`chcp 65001` is an alternative.)

## Step 1 — start the local judge

Only needed for judge-layer tests, `test_first_eval.py`, and
`test_goldens_from_scratch.py`. Skip it for offline-only runs (see
"Running without the judge" below) or if `.env` points at hosted Jev.

```shell
.venv/Scripts/python run_judge.py
```

This serves Laya over DeepEval's Jev wire protocol on
`http://127.0.0.1:8000`. **Leave this terminal open** — the tests need the
server running.

Verify it is up (in a second terminal):

```shell
curl http://127.0.0.1:8000/health
```

Expect `{"status":"ok","loaded":["typed-decisions"],...}`.

> First-run note: the checkpoint loads lazily, so the first judge request
> can take a minute or two while `convaiinnovations/laya` downloads/loads.
> Every later request answers in ~1–3s. If a test times out at ~88s right
> after starting the server, wait for `loaded` to appear in `/health` and
> rerun — it is a cold start, not a test failure.

## Step 2 — smoke test the judge connection

```shell
# Git Bash
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_first_eval.py

# PowerShell
$env:PYTHONUTF8="1"; .venv\Scripts\deepeval.exe test run test_first_eval.py
```

Expected: `1 passed` in a few seconds (warm judge). If this fails with
connection errors, the judge isn't up — fix that first.

## Step 3 — run everything

`deepeval test run` takes a file **or directory**, and forwards extra
pytest args. To run the whole repo's tests in one command:

```shell
# Git Bash
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run . --ignore=_demo

# PowerShell
$env:PYTHONUTF8="1"; .venv\Scripts\deepeval.exe test run . --ignore=_demo
```

The `--ignore=_demo` is required: `_demo/` is a snapshot copy of the
project, and its duplicate `test_*.py` basenames break pytest collection
("import file mismatch"). With it, all **144 tests** are collected.

Equivalently, list the files you want — extra positional args pass through
to pytest:

```shell
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_first_eval.py test_agent.py test_goldens_from_scratch.py
```

Or run each file separately — useful because they need different things:

```shell
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_first_eval.py            # 1 test,   judge needed
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent.py                 # 80 tests, judge for judge layer only
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent_extended.py        # 53 tests, judge for grounding only
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_goldens_from_scratch.py  # 9 tests,  judge needed (see caveats)
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_first_eval_geval.py      # 1 test,   OPENAI_API_KEY needed
```

Expected results per file (default local-Laya setup):

| File | Expected |
|---|---|
| `test_first_eval.py` | 1 passed, a few seconds |
| `test_agent.py` | `72 total tests, 100% passed` in ~3–4 min (the judge phase dominates). pytest collects 80 items; the 8 `test_query_group_complete` checks run as plain pytest asserts and aren't counted in DeepEval's table. |
| `test_agent_extended.py` | `49 total tests, 100% passed` in ~3.5 min (grounding calls dominate). pytest collects 53 items: 2 multi-turn checks are plain asserts, 2 are skipped unless `AGENT_BACKEND=llm`. |
| `test_goldens_from_scratch.py` | **Expect failures under local Laya** — see its section. Should pass 9/9 against hosted `jev-1.13`. |
| `test_first_eval_geval.py` | 1 passed if `OPENAI_API_KEY` is set; errors without it. |

Results are written to `deepeval_results/test_run_*.json`
(`DEEPEVAL_RESULTS_FOLDER`).

## test_agent.py — the layered suite

Each query q1–q8 gets goldens in every layer, run cheapest first.
`SEQUENCE_FAIL_FAST = True` means once a layer fails for a query, that
query's later layers are skipped — a failure early shows up as skipped
tests downstream, not extra failures.

Per query, the tests are (72 eval cases + 8 data checks total):

| Test id pattern | Count | What it asserts |
|---|---|---|
| `test_query_group_complete[qN]` | 8 | Golden-set integrity: each query has exactly one pass golden plus one per failure mode the layer owns. Pure data check — instant, no agent, no judge. |
| `test_eval[qN-structured-pass]` | 8 | The live agent's `search_knowledge` calls obey the tool contract (1–5 single-word keywords; output is `N paragraphs matched; showing M.` + exactly M blocks; M ≤ N and M ≤ `agent.MAX_SEARCH_RESULTS`; every returned paragraph contains a keyword) and `ToolCorrectnessMetric` matches `expected_tools`. |
| `test_eval[qN-structured-fail_invalid_keywords]` | 8 | Mocked agent calls the tool with malformed keywords → validation must fail; tool correctness still passes (a call was made). |
| `test_eval[qN-structured-fail_irrelevant_results]` | 8 | Mocked agent gets back paragraphs containing none of the keywords → validation must fail. |
| `test_eval[qN-structured-fail_no_tool_call]` | 8 | Mocked agent never calls `search_knowledge` → `ToolCorrectnessMetric` must fail (per-call validation can't see this). |
| `test_eval[qN-fuzzy-pass]` | 8 | Live answer meets ≥60% of the key-phrase requirements (exact → RapidFuzz ≥85 → stemmed BLEU/ROUGE ≥0.6 cascade). |
| `test_eval[qN-fuzzy-fail_incomplete]` | 8 | Canned thin-but-truthful answer → coverage must fail. This is where incomplete answers are gated. |
| `test_eval[qN-judge-pass]` | 8 | Live answer: "No contradiction" (min over per-sentence agree-probabilities) and "Completeness" (4-level Score) both pass vs. the reference answer. |
| `test_eval[qN-judge-fail_contradiction]` | 8 | Canned fluent answer containing one false claim → "No contradiction" must catch it. |
| `test_eval[qN-judge-fail_incomplete]` | 8 | Canned thin answer → no contradiction expected; completeness is scored but report-only (`expected_pass=None`) because laya's score head reads topical overlap, not coverage. |

### Filtering with `-k`

`-k` isn't listed in `deepeval test run --help`, but it's forwarded to
pytest and does substring/expression matching on test ids, which are
`qN-LAYER-SCENARIO` (plus `qN` on the group checks).

| Command suffix | Selects | Count | Needs judge? |
|---|---|---|---|
| `-k "structured"` | whole structured layer | 32 | no |
| `-k "fuzzy"` | whole fuzzy layer | 16 | no |
| `-k "judge"` | whole judge layer | 24 | yes |
| `-k "query_group_complete"` | golden-set integrity checks | 8 | no |
| `-k "not judge"` | structured + fuzzy + checks — **fully offline** | 56 | no |
| `-k "q3"` | every test for query 3 (all layers + its check) | 10 | partly |
| `-k "pass"` | the live-agent pass goldens in every layer | 24 | partly |
| `-k "fail"` | every planted-failure test in every layer | 48 | partly |
| `-k "fail_invalid_keywords"` / `-k "fail_irrelevant_results"` / `-k "fail_no_tool_call"` | one structured failure mode, all queries | 8 each | no |
| `-k "fail_incomplete"` | thin-answer tests (8 fuzzy + 8 judge) | 16 | partly |
| `-k "fail_contradiction"` | planted contradictions | 8 | yes |
| `-k "q3 and fuzzy"` | query 3's fuzzy tests | 2 | no |
| `-k "structured and fail"` | all planted structured failures | 24 | no |
| `-k "q3-fuzzy-pass"` | one exact test | 1 | no |

Examples:

```shell
# one parametrized case
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent.py -k "q3-fuzzy-pass"

# a whole layer
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent.py -k "judge"

# everything that doesn't need the judge (runs in seconds, fully offline)
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent.py -k "not judge"
```

### Running without the judge

The structured layer, fuzzy layer, and the group checks never touch the
judge — with `AGENT_BACKEND=extractive` they need no network at all:

```shell
PYTHONUTF8=1 .venv/Scripts/deepeval.exe test run test_agent.py -k "not judge"   # 56 tests, ~4s
```

### Layer toggles

Module constants at the top of `test_agent.py`:
`RUN_STRUCTURED_EVALS`, `RUN_FUZZY_EVALS`, `RUN_JEV_EVALS` enable/disable
whole layers at load time; `SEQUENCE_FAIL_FAST` controls the skip
behaviour. Prefer `-k` for ad-hoc subsets and the toggles for persistent
changes (e.g. a CI job with no judge).

## test_agent_extended.py — the behavioural extension

53 tests probing *how the agent behaves*, not just what it outputs. Same
self-verification style: canned bad actors are expected to be *caught*.
Only `test_grounding*` needs the judge; everything else is local. Test ids
reuse the `qN-scenario` pattern (`q9`/`q10` = out-of-scope refusals, `q11`
= the multi-part query), so `-k` filters work the same way
(`-k "trajectory"`, `-k "refusal"`, `-k "not grounding"` for fully offline).

| Group | Tests | What it checks |
|---|---|---|
| `test_trajectory_and_keywords[qN-pass]` | 9 | ≥1 search before answering, ≤4 tool calls, converged inside `MAX_STEPS`; searched keywords grounded in the query/context |
| `test_trajectory_flail_caught`, `test_lazy_refusal_caught`, `test_keyword_relevance_irrelevant_caught` | 3 | Canned bad behaviour must be caught (over-budget calls, refusing without searching, on-contract but irrelevant keywords) |
| `test_out_of_scope_refusal` / `..._answered_anyway_caught` | 3 | Exact `"Sorry I cannot help with that."` for out-of-scope queries — but only *after* searching; a confident answer instead must fail |
| `test_grounding` / `test_grounding_catches_unsupported` | 25 | Faithfulness: each answer sentence is judged against its best-matching retrieved paragraph (one paragraph per sentence — the checkpoint misses contradictions against large context states); canned contradictions must fail, thin-but-true stays grounded |
| `test_answer_stability[qN-pass]` | 9 | Same query ×3 runs: identical under `extractive`, ≥85% similar under `llm` |
| `test_multi_turn_*` | 3 | History accumulates; capped at `MAX_USER_TURNS`; pronoun follow-up (llm-gated) |
| `test_llm_backend_smoke` | 1 | Real-model tool-choice smoke test (llm-gated) |

New data in `goldens/agent_extended_goldens.json`. Note: refusal queries
were chosen so every keyword *stem* (≥4 chars) misses the knowledge base —
under `extractive`, a stem collision (e.g. "bread"→"brea"→"break") retrieves
paragraphs and the agent answers instead of refusing. And four tests skip
under plain `pytest` — they need `deepeval test run`'s per-test trace.

## test_goldens_from_scratch.py — tutorial-04 artifact

9 tests over `goldens/ai_engineering_goldens.json` (3 questions ×
pass/fail_contradiction/fail_incomplete), one judge call per test
reporting two metrics: "No contradiction" (a `Noul`) and "Completeness"
(a `Score`).

**Under the local `typed-decisions` checkpoint, expect failures** — this
is a checkpoint limitation, not a harness bug:

- It asks contradiction as a `Noul`. Laya's noul head reads the
  `false:`/`true:` label token instead of the state (laya issue #156), so
  its verdicts come out wrong. The main suite works around this by using
  two-option `Choice`s; this older file predates that fix.
- Its `fail_incomplete` tests expect the `Score` head to grade coverage.
  `typed-decisions` scores topical overlap instead, rating thin on-topic
  answers ~0.7, so those tests fail even though the harness is fine.

Against hosted `jev-1.13` (three-line `.env` change), where both heads
behave, it should pass 9/9. Filter it like the others —
`-k "fail_contradiction"`, `-k "q2"`, etc.

## test_first_eval_geval.py — the OpenAI comparison

The same context-window eval as `test_first_eval.py`, but judged by
`GEval` — a generative LLM-as-judge that prompts a model to grade the
answer. It's the contrast case for the decision-model approach: prose +
parsing instead of states + probabilities. Does not need the local judge.

The judge LLM doesn't have to be OpenAI — `GEval` resolves a default model
from deepeval's provider settings, and any supported provider works,
including local models:

- **OpenAI** (default): `OPENAI_API_KEY=<key>` in `.env`.
- **Ollama** (e.g. qwen): either `deepeval set-ollama --model qwen3`, or
  `.env` vars — `LOCAL_MODEL_API_KEY=ollama` (the literal sentinel that
  selects Ollama), `OLLAMA_MODEL_NAME=qwen3`,
  `LOCAL_MODEL_BASE_URL=http://localhost:11434`. Ollama must be running
  with the model pulled.
- **Any OpenAI-compatible server** (LM Studio, vLLM, Ollama's `/v1`):
  `deepeval set-local-model --model <name> --base-url <url>` or
  `USE_LOCAL_MODEL=yes` + `LOCAL_MODEL_NAME` / `LOCAL_MODEL_BASE_URL` /
  `LOCAL_MODEL_API_KEY`.
- **Explicit in code**: `GEval(model=OllamaModel(model="qwen3"))` or any
  `DeepEvalBaseLLM` subclass — no env needed.

Expect less consistent scores from small local models — GEval makes the
model generate its own grading steps, which weak models do unreliably.

## Useful `deepeval test run` flags

Own flags (see `deepeval test run --help`):

| Flag | Effect |
|---|---|
| `-x` | stop on first failure |
| `-v` | verbose output |
| `-r N` | repeat each test N times (mind the judge cost) |
| `-d failing` | show only failures in the end-of-run table |
| `-id NAME` | label the run (appears in the results file) |
| `-c` | reuse cached results instead of re-evaluating |
| `-s` | skip test cases with missing parameters |

Anything else is forwarded to pytest: `-k` (above), `--collect-only`,
`--ignore=...`, `-m`, `--durations=N`.

Avoid `-n` (pytest-xdist) with `test_agent.py`: the fail-fast sequencing
keeps per-query state in memory and pytest distributes tests across
processes with no shared state or ordering, so skips won't propagate.

## Running from Windows Command Prompt (cmd.exe)

The same steps work in `cmd.exe` — only the path separators and the way
you set `PYTHONUTF8` differ. `set` lasts for the whole window, so you only
need it once instead of prefixing every command.

```bat
REM once per window: lets the results table print its Unicode marks
set PYTHONUTF8=1

REM window 1 -- start the judge and leave it running (skip for offline-only runs)
.venv\Scripts\python run_judge.py

REM window 2 -- health check (curl.exe ships with Windows 10/11)
curl http://127.0.0.1:8000/health

REM same window -- smoke test, then the test files
.venv\Scripts\deepeval.exe test run test_first_eval.py
.venv\Scripts\deepeval.exe test run test_agent.py
.venv\Scripts\deepeval.exe test run test_goldens_from_scratch.py
.venv\Scripts\deepeval.exe test run test_first_eval_geval.py   REM needs an LLM judge (OPENAI_API_KEY or set-ollama)
.venv\Scripts\deepeval.exe test run test_agent_extended.py     REM judge needed for grounding tests only

REM or everything at once (--ignore=_demo avoids duplicate-basename errors)
.venv\Scripts\deepeval.exe test run . --ignore=_demo

REM offline only, no judge needed
.venv\Scripts\deepeval.exe test run test_agent.py -k "not judge"

REM subsets work the same way
.venv\Scripts\deepeval.exe test run test_agent.py -k "judge"
.venv\Scripts\deepeval.exe test run test_agent.py -k "structured"
.venv\Scripts\deepeval.exe test run test_agent.py -k "q3-fuzzy-pass"
.venv\Scripts\deepeval.exe test run test_agent.py -k "fail_incomplete"
```

Notes:

- If `.env` doesn't exist yet: `copy .env.example .env`.
- Use `set`, not `setx` — `setx` writes the variable permanently and only
  applies to windows opened afterwards.
- `chcp 65001` is an alternative to `set PYTHONUTF8=1` for the Unicode
  output issue; either one works.

## Troubleshooting

- **`UnicodeEncodeError` at the end of the run (Windows):** the results
  table can't print `⚠`/`✓` through a cp1252 console. Always run with
  `PYTHONUTF8=1` as shown above (or `chcp 65001` first).
- **`httpcore2.ReadTimeout` / `RetryError` on the first judge call:** the
  checkpoint is still loading. Check `/health` and rerun once `loaded`
  includes `typed-decisions`.
- **`import file mismatch` collecting `test_*.py`:** you're running
  against a directory and pytest found `_demo/`'s duplicate test files.
  Add `--ignore=_demo` or name the files explicitly.
- **`OPENAI_API_KEY` errors on `test_first_eval_geval.py` or `generate_*.py`:**
  expected — those need a real OpenAI key in `.env`; skip them for the
  offline setup.
- **`test_goldens_from_scratch.py` fails under the local judge:**
  expected — it uses `Noul` (broken on `typed-decisions`, laya #156) and
  relies on the score head grading coverage (it reads topical overlap
  instead). It passes against hosted `jev-1.13`.
- **Extended tests skip under plain `pytest`:** trajectory/refusal/
  grounding/llm tests read the per-test trace that only `deepeval test run`
  creates. Run them through `deepeval test run`, not `pytest`.
- **`RetryError`/`TimeoutError` on grounding-style judge calls:** the judge
  request timed out — keep context states small. `GroundingMetric` sends
  one retrieved paragraph per sentence group for exactly this reason; don't
  pass whole tool outputs as `CONTEXT`/`expected_output` states.
- **An out-of-scope refusal test answers instead of refusing:** under
  `extractive`, a keyword *stem* matched the KB (e.g. "bread"→"brea"→"break").
  Pick queries whose keywords and ≥4-char stems are all absent from
  `agent-knowledge.md`.
- **portalocker/shared-lock warnings:** benign on Windows unless
  `portalocker[win32]` is installed; result caching still works.
- **"shows N results, limit is M" structured failure:** the validator reads
  `agent.MAX_SEARCH_RESULTS`; it fails only if agent output exceeds that
  constant, so keep the two in sync.
- **Judge server stopped:** tests that need it fail with connection
  errors, not eval failures. Restart `run_judge.py` — or run
  `-k "not judge"` while it's down.
- **A `llm`-backend live test fails:** with `AGENT_BACKEND=llm` the
  pass-golden tests depend on the real model's behaviour — check
  `INFERENCE_*` settings and run `agent.py` interactively to see the
  answer it produced.

## Why the judge questions look unusual

- Contradiction is asked per **sentence** of the answer (`min` over the
  per-sentence verdicts). A whole-answer judgement lets fluent answers with
  one blatant false claim slip through on this checkpoint.
- The questions are two-option `Choice`s ("agrees"/"contradicts"), not
  `Noul`: laya's shipped noul head reads the `false:`/`true:` label token
  instead of the state (laya issue #156).
- `fail_incomplete` goldens only require "No contradiction" — laya's score
  head can't grade coverage reliably, so completeness is reported but not
  pass/fail for that scenario. The fuzzy layer gates thin answers instead.
