# DeepEval agent evals

The demo repo for a nine-part tutorial series on evaluating an AI agent with
[DeepEval](https://deepeval.com/docs/getting-started).

The subject under test is a small ReAct agent ([agent.py](agent.py)) that answers
beginner questions about AI engineering from a Markdown knowledge base
([agent-knowledge.md](agent-knowledge.md)). The series builds layered evals for it:
structured validation of tool calls, fuzzy matching of answers, and a judge.
On top of the tutorial suite, `test_agent_extended.py` adds behavioural evals —
trajectory/step budgets, keyword relevance, the refusal path, grounding vs.
retrieved context, multi-turn memory, and answer stability — see
[explanation.md](explanation.md) for what every test does and why, and
[how-to-test.md](how-to-test.md) for the runbook.

## Quick start

```shell
uv sync
cp .env.example .env   # then fill in your keys
uv run agent.py
```

## Tags

Each tutorial has a git tag holding its finished state. To start a tutorial,
check out the previous tutorial's tag:

| Tag | Tutorial |
| --- | --- |
| `tutorial-01` | Setting up the repo and exploring the agent |
| `tutorial-02` | Understanding layered agent evals |
| `tutorial-03` | Writing your first eval with a judge |
| `tutorial-04` | Generating goldens and testing expected failures |
| `tutorial-05` | Generating goldens from a knowledge base |
| `tutorial-06` | Evaluating tool calls with structured validation |
| `tutorial-07` | Evaluating agent answers with fuzzy matching |
| `tutorial-08` | Evaluating agent answers with a judge |
| `tutorial-09` | Running layered evals that fail fast |

For example, to start Tutorial 3:

```shell
git checkout tutorial-02
```
