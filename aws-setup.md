# Setup on AWS EC2

A copy-paste runbook for running this repo's eval suite on a fresh EC2
instance. Everything runs on the instance itself — the agent uses the
deterministic `extractive` backend and the judge is a local Laya server —
so **no API keys are needed** and the only inbound port required is SSH.

**Recommended setup: Ubuntu 24.04 LTS on a `t3.large`.** Ubuntu's default
`python3` is 3.12, which works out of the box — Amazon Linux 2023 ships
Python 3.9 (too old) so you have to install `python3.11` explicitly. Both
are covered below, but pick Ubuntu for the simplest demo.

## What you'll end up with

- The judge (`laya-serve`) running in a `tmux` session on
  `127.0.0.1:8000`.
- The full test suite passing: `72 total tests, 100% passed` on
  `test_agent.py` plus `49 total tests, 100% passed` on the behavioural
  extension `test_agent_extended.py`.
- Roughly 30 minutes including the one-time ~850 MB model download.

## Step 1 — launch the instance

In the EC2 console, **Launch instance** with:

| Setting | Value |
| --- | --- |
| AMI | **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type** — x86_64 |
| Instance type | `t3.large` (2 vCPU / 8 GiB). `t3.medium` (4 GiB) is the minimum — see §8 for the swap you'll need. ~$0.08/hour for t3.large on-demand. |
| Key pair | one you have the `.pem` file for |
| Security group | inbound 22 (SSH) from your IP only |
| Root volume | 30 GiB gp3 (torch + the ~850 MB checkpoint + venv ≈ 8–10 GiB) |
| Outbound | default "allow all" (needed for PyPI and the one-time checkpoint download from Hugging Face) |

Do **not** open port 8000 inbound — the tests reach the judge over
loopback.

## Step 2 — SSH in

```shell
ssh -i /path/to/your-key.pem ubuntu@<instance-public-ip>
```

Use `ubuntu@` on Ubuntu (or `ec2-user@` on Amazon Linux). Everything below
runs on the instance.

## Step 3 — install system packages

```shell
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git tmux curl
```

Ubuntu 24.04 gives you Python 3.12 — any Python 3.11+ works (the dev venv
for this repo runs 3.13). Check with `python3 --version`.

## Step 4 — get the code and create the virtualenv

Clone the repo, or if it isn't in a remote, copy it up from your machine:

```shell
git clone <repo-url> jeveval
# or, from your local machine instead:
# scp -i /path/to/your-key.pem -r ./jeveval ubuntu@<instance-public-ip>:~/

cd jeveval
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Step 5 — install Python dependencies

```shell
pip install -r requirements.txt
```

That installs everything: `deepeval`, `laya[serve]` (the judge server),
and a CPU-only `torch` — `requirements.txt` points pip at PyTorch's CPU
index so you don't get the multi-GB CUDA build.

Validate:

```shell
python -c "import deepeval, laya, torch; print('ok')"
```

You should see `ok`.

## Step 6 — configure .env

```shell
cp .env.example .env
```

The example defaults already point at the local judge and the
`extractive` agent — **nothing needs editing** for a demo run.

## Step 7 — start the judge (leave it running)

Run it in `tmux` so it survives your SSH session:

```shell
tmux new -s judge
python run_judge.py
```

Then detach with `Ctrl-b` then `d` — the server keeps running. Reattach
later with `tmux attach -t judge`.

Prefer no tmux? `nohup python run_judge.py > judge.log 2>&1 &` works too.

Validate (in a second SSH session, or the same one after `nohup`):

```shell
curl http://127.0.0.1:8000/health
```

You should see `{"status":"ok","loaded":[...],...}`. Right after startup
`loaded` may be empty — the `typed-decisions` checkpoint downloads
(~850 MB) and loads on the first judge request, which is what the smoke
test below triggers. Wait until `loaded` includes `typed-decisions`
before running the full suite.

## Step 8 — run the tests

```shell
PYTHONUTF8=1 .venv/bin/deepeval test run test_first_eval.py      # smoke test
PYTHONUTF8=1 .venv/bin/deepeval test run test_agent.py           # main suite
PYTHONUTF8=1 .venv/bin/deepeval test run test_agent_extended.py  # behavioural extension
```

Expected:

- Smoke test: `1 passed`. **First run may take 1–2 minutes or hit a ~88s
  `ReadTimeout` while the checkpoint downloads — that's a cold start, not
  a failure.** Check `/health`, wait for `loaded` to include
  `typed-decisions`, and rerun.
- Main suite: `72 total tests, 100% passed`. Budget ~5–8 minutes — the
  judge phase dominates on 2 vCPUs (a laptop does it in ~3–4).
- Extension: `49 total tests, 100% passed`, plus 2 plain pytest checks and
  2 `llm`-gated skips — a similar budget; the grounding calls dominate.
- Results land in `deepeval_results/test_run_*.json`.

Quick validation that the offline layers work without any judge
dependency (useful while the checkpoint downloads):

```shell
PYTHONUTF8=1 .venv/bin/deepeval test run test_agent.py -k "not judge"   # 48 evals, seconds
```

The repo's other two test files are deliberately **not** part of this run —
if you run them (or `deepeval test run .`), expect failures for documented
reasons, not a broken setup: `test_goldens_from_scratch.py` (9 tests) uses
the `Noul` question form and coverage `Score`s the local checkpoint gets
wrong — it only passes against hosted `jev-1.13` — and
`test_first_eval_geval.py` (1 test) needs an LLM judge (`OPENAI_API_KEY`
or a local Ollama model). See [how-to-test.md](how-to-test.md) for every
file and `-k` filter — layer subsets, per-query, per-failure-mode — plus
what each layer checks.

## Step 9 — done? stop the instance

The suite keeps no state between runs, so a stopped instance only costs
its EBS storage. Stop the judge first if you like (`tmux kill-session -t
judge`), then stop the instance from the EC2 console.

## Amazon Linux 2023 variant

Identical flow; only the package step differs — AL2023's default
`python3` is 3.9, so install 3.11 explicitly:

```shell
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip git tmux
python3.11 -m venv .venv        # in step 4, not `python3 -m venv`
```

SSH user is `ec2-user@`. Everything else is the same.

## Troubleshooting

- **`ReadTimeout` / `RetryError` on the first judge call:** the
  checkpoint is still downloading/loading. Check `/health` and rerun once
  `loaded` includes `typed-decisions`.
- **Judge process died / `curl` refused:** reattach (`tmux attach -t
  judge`) or rerun `python run_judge.py`. With `nohup`, check `judge.log`.
- **Killed/OOM on `t3.medium` (4 GiB):** add swap so the checkpoint load
  can't OOM:

  ```shell
  sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  ```

  (`swapon` doesn't survive a reboot — rerun `sudo swapon /swapfile`, or
  add an `/etc/fstab` line, if you restart the instance.)

- **Disk full:** the CUDA build of torch is the usual suspect —
  reinstall CPU-only with `pip install torch --index-url
  https://download.pytorch.org/whl/cpu`.
- **`ModuleNotFoundError` in tests:** you're outside the venv —
  `source .venv/bin/activate`, or call `.venv/bin/...` paths directly as
  §8 does.
- **Remote judge (not needed for the demo):** if you ever want the judge
  reachable from another host, set `LAYA_HOST=0.0.0.0` before starting
  `run_judge.py` and scope the security-group rule to that host only.
