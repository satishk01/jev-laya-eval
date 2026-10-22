"""Start the local judge: Laya served over the Jev /v1/systemone wire protocol.

`laya-serve` exposes the same endpoint DeepEval's JevEval calls through the
TypeSafe SDK, so the eval code is identical whether the backend is the hosted
Jev API or this local Laya server. Which one answers is decided entirely by
TYPESAFE_BASE_URL / TYPESAFE_API_KEY / TYPESAFE_MODEL_NAME in .env:

- Laya (this server):  TYPESAFE_BASE_URL=http://localhost:8000,
  TYPESAFE_API_KEY=<any non-empty value>, TYPESAFE_MODEL_NAME=typed-decisions
  (or english / multilingual).
- Jev (hosted):        TYPESAFE_BASE_URL=<your gateway>,
  TYPESAFE_API_KEY=<your key>, TYPESAFE_MODEL_NAME=jev-1.13.

Usage:
    .venv/Scripts/python run_judge.py        # http://localhost:8000
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Only the checkpoint TYPESAFE_MODEL_NAME pins needs to be resident; loading
# just it keeps startup and memory down on a laptop. Everything else Laya
# serves lazily on first request anyway.
os.environ.setdefault("LAYA_HOST", "127.0.0.1")
os.environ.setdefault("LAYA_PORT", "8000")
os.environ.setdefault("LAYA_PRELOAD", "0")

from laya.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
