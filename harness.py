#!/usr/bin/env python3
"""Pipe check: send one fixed prompt to each inference provider and record the result.

Usage:
    python harness.py                    # all providers
    python harness.py --only fireworks   # one or more providers

Keys are read from environment variables (or a .env file next to this script).
Each run writes runs/<timestamp>.json. Keys are sent only in the Authorization
header and are never written to the run file.

Exit code: 0 if every provider PASSes, 1 otherwise (FAIL or SKIP).
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

import httpx

# ---- Providers: one row per backend. Model ids differ by provider on purpose;
# ---- see the README note about pipe checks vs. comparisons.
PROVIDERS = {
    "baseten": {
        "url": "https://inference.baseten.co/v1/chat/completions",
        "key_env": "BASETEN_API_KEY",
        "model": "zai-org/GLM-5.3",
    },
    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "key_env": "DEEPINFRA_API_KEY",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    },
    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "key_env": "FIREWORKS_API_KEY",
        "model": "accounts/fireworks/models/glm-5p3-flash",
    },
}

# ---- The case: identical for every provider.
CASE = {
    "name": "pipe-check-pong",
    "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
    "temperature": 0,
    # Generous so reasoning-style models do not hit the cap before answering.
    "max_tokens": 512,
    "expect_substring": "pong",
}

TIMEOUT_S = 60.0
RUNS_DIR = pathlib.Path(__file__).resolve().parent / "runs"
ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"


def load_dotenv(path):
    """Minimal KEY=VALUE loader. Existing environment variables win."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def call_provider(name, cfg, case, client):
    """Make one request and return a record. Never raises."""
    record = {
        "provider": name,
        "url": cfg["url"],
        "model": cfg["model"],
        "request": {
            "model": cfg["model"],
            "messages": case["messages"],
            "temperature": case["temperature"],
            "max_tokens": case["max_tokens"],
        },
        "status": None,
        "elapsed_ms": None,
        "finish_reason": None,
        "usage": None,
        "reply": None,
        "error": None,
        "verdict": "FAIL",
    }

    key = os.environ.get(cfg["key_env"], "")
    if not key:
        record["verdict"] = "SKIP"
        record["error"] = f"missing environment variable {cfg['key_env']}"
        return record

    started = time.perf_counter()
    try:
        resp = client.post(
            cfg["url"],
            json=record["request"],
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    record["status"] = resp.status_code

    try:
        data = resp.json()
    except ValueError:
        record["error"] = f"non-JSON body: {resp.text[:500]}"
        return record

    if resp.status_code != 200:
        record["error"] = json.dumps(data)[:1000]
        return record

    try:
        choice = data["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        record["usage"] = data.get("usage")
        reply = (choice.get("message") or {}).get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        record["error"] = f"unexpected response shape: {exc}"
        return record

    if reply is None:
        record["error"] = "no message.content in response (check finish_reason)"
        return record

    record["reply"] = reply
    if case["expect_substring"].lower() in reply.lower():
        record["verdict"] = "PASS"
    else:
        record["error"] = f"reply does not contain {case['expect_substring']!r}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only", nargs="+", choices=sorted(PROVIDERS), help="run only these providers"
    )
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    names = args.only or list(PROVIDERS)
    started = datetime.datetime.now(datetime.timezone.utc)

    records = []
    with httpx.Client() as client:
        for name in names:
            rec = call_provider(name, PROVIDERS[name], CASE, client)
            records.append(rec)
            detail = rec["reply"] if rec["reply"] is not None else rec["error"]
            status = "-" if rec["status"] is None else str(rec["status"])
            elapsed = "-" if rec["elapsed_ms"] is None else f"{rec['elapsed_ms']}ms"
            print(f"{rec['verdict']:<5}{name:<11}{status:<5}{elapsed:<9}{detail!r}")

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out = {
        "started_utc": started.isoformat(),
        "python": sys.version.split()[0],
        "httpx": httpx.__version__,
        "case": CASE,
        "records": records,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path.relative_to(pathlib.Path(__file__).resolve().parent)}")

    all_pass = all(r["verdict"] == "PASS" for r in records)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
