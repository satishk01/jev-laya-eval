# Jev/Laya agent evals

An evaluation suite for [`agent.py`](agent.py) — a small ReAct agent that
answers beginner AI-engineering questions by searching a Markdown knowledge
base ([`agent-knowledge.md`](agent-knowledge.md)) with a single tool,
`search_knowledge(keywords)`.

The suite checks the agent in layers, cheapest first:

1. **Structured** — rule-based validation of the tool calls recorded in each
   run's DeepEval trace (`deepeval_tool_util.py`, `ToolCorrectnessMetric`)
2. **Fuzzy** — key-phrase coverage of the answer via an exact → RapidFuzz →
   stemmed BLEU/ROUGE cascade (`deepeval_fuzzy_util.py`)
3. **Judge** — a *decision-model* judge verifies each answer sentence against
   a reference answer. Runs against **Laya** (open-weight checkpoint served
   locally by `run_judge.py`) or hosted **Jev** — same code, chosen purely by
   `.env`.

plus a **behavioural extension** (`test_agent_extended.py`) — trajectory/step
budgets, keyword relevance, the out-of-scope refusal path, grounding vs.
retrieved paragraphs, multi-turn memory, and answer stability.

Every layer is fed planted-failure goldens it must *catch*, so the suite also
verifies the eval harness itself. 144 tests across five files.

## Quick start

```shell
# 1. environment — pip or uv both work
python -m venv .venv
source .venv/Scripts/activate   # Git Bash (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
#   ...or: uv sync

# 2. config — the defaults need no API keys
cp .env.example .env            # extractive agent + local Laya judge

# 3. start the judge (terminal 1, leave running)
python run_judge.py             # serves Laya on http://127.0.0.1:8000
curl http://127.0.0.1:8000/health   # wait for "loaded": ["typed-decisions"]

# 4. run the suite (terminal 2)
PYTHONUTF8=1 deepeval test run test_agent.py           # layered suite
PYTHONUTF8=1 deepeval test run test_agent_extended.py  # behavioural extension
```

Expected: `72 total tests, 100% passed` and `49 total tests, 100% passed`
respectively (a few minutes — the judge calls dominate). To run the agent
interactively instead: `python agent.py` (`/exit` to quit).

Fully offline subset, no judge needed:

```shell
PYTHONUTF8=1 deepeval test run test_agent.py -k "not judge"
```

## Documentation

| File | What it covers |
|---|---|
| [explanation.md](explanation.md) | What every test does and why — start here |
| [how-to-test.md](how-to-test.md) | The runbook: commands, `-k` filters, troubleshooting |
| [jev-eval-technical-guide.md](jev-eval-technical-guide.md) | Deep dive on the Jev/Laya judge mechanics and the layered design |
| [aws-setup.md](aws-setup.md) | Copy-paste runbook for a fresh EC2 instance |

## Repo layout

| File | Role |
|---|---|
| `agent.py` | System under test: ReAct loop, `search_knowledge` tool, `llm` and deterministic `extractive` backends |
| `run_judge.py` | Launches the local Laya judge (Jev `/v1/systemone` wire protocol) |
| `test_agent.py` | The layered suite: structured → fuzzy → judge, fail-fast per query |
| `test_agent_extended.py` | Behavioural extension: trajectory, keyword quality, refusal, grounding, multi-turn, stability |
| `test_first_eval.py` / `test_first_eval_geval.py` | Judge-connection smoke tests (JevEval and GEval comparison) |
| `test_goldens_from_scratch.py` | Flat judge suite over synthesized goldens (known-failing under local Laya — see how-to-test.md) |
| `deepeval_{tool,fuzzy,jev}_util.py` | The custom metrics and trace plumbing |
| `generate_*.py` | Golden-set generators (committed outputs in `goldens/`) |
| `.env.example` | Judge backend (`TYPESAFE_*`), agent backend (`AGENT_BACKEND`, `INFERENCE_*`) |
