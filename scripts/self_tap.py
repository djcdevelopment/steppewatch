#!/usr/bin/env python3
"""One-command Agent Self-Tap entrypoint for Steppewatch.

Designed for frontier agents (Claude Code, Codex CLI, Hermes, etc.) and human
builders to verify local silicon, discover active local transcripts, validate
thrash separation, and emit machine-readable JSON onboarding commands in <1 s.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from local_embedder import LocalEmbedder, detect_agent_thrash, probe_hardware  # noqa: E402
from importlib import import_module  # noqa: E402


def discover_transcripts_safe(max_files: int = 8, hermes_home: Path | None = None) -> List[Dict[str, Any]]:
    watcher = import_module("02_watch_frontier_agents")
    items = watcher.discover_recent_transcripts(max_files=max_files, hermes_home=hermes_home)
    out: List[Dict[str, Any]] = []
    for source, path in items:
        try:
            st = (path.database if isinstance(path, watcher.DatabaseSession) else path).stat()
            out.append(
                {
                    "source": source,
                    "path": str(path),
                    "bytes": st.st_size,
                    "age_seconds": round(max(0.0, time.time() - st.st_mtime), 1),
                }
            )
        except OSError:
            continue
    return out


def run_fast_self_check(embedder: LocalEmbedder) -> Dict[str, Any]:
    healthy = detect_agent_thrash(
        embedder,
        [
            "Step 1: Inspected webhook_client.py send_event() to verify current retry backoff and timeout defaults.",
            "Step 2: Edited webhook_client.py to add max_timeout_sec=15.0 and fixed TypeError on RetryConfig init.",
            "Step 3: Updated test_webhook_client.py to add test_retry_timeout_guard covering HTTP 504 responses.",
            "Step 4: Ran pytest test_webhook_client.py; caught AssertionError at line 88 and patched backoff jitter.",
            "Step 5: Re-ran pytest test_webhook_client.py; all 12 tests passed with exit code 0.",
        ],
        trace_id="self-tap-healthy",
    )
    stall = detect_agent_thrash(
        embedder,
        [
            "Step 1: Let me carefully review the webhook integration requirements again to make sure I have the full picture before proposing an edit.",
            "Step 2: I will re-examine the webhook integration plan and verify how the retry timeout should be structured before making any code edits.",
            "Step 3: Let me double-check the webhook integration architecture one more time so the proposed design aligns with existing client boundaries.",
            "Step 4: Before writing any code, I want to carefully reconsider the webhook integration approach and review the timeout constraints again.",
            "Step 5: I am revisiting the webhook integration design notes again to confirm the retry timeout details before deciding on the next edit.",
        ],
        trace_id="self-tap-stall",
    )
    ok = (not healthy["thrash_detected"]) and bool(stall["thrash_detected"])
    return {
        "ok": ok,
        "healthy_score": healthy["thrash_score"],
        "stall_score": stall["thrash_score"],
        "latency_ms": round(healthy["latency_ms"] + stall["latency_ms"], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Steppewatch agent self-tap probe (JSON output).")
    parser.add_argument("--hermes-home", type=Path, default=None, help="Hermes root containing sessions, state.db and profiles (default: ~/.hermes)")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Target workspace repository path")
    parser.add_argument("--force-hash", action="store_true", help="Use zero-download hash384 embedder")
    args = parser.parse_args()

    hw = probe_hardware()
    embedder = LocalEmbedder(force_hash=args.force_hash)
    self_check = run_fast_self_check(embedder)
    transcripts = discover_transcripts_safe(max_files=8, hermes_home=args.hermes_home)

    repo_str = str(args.repo.resolve())
    episodes_path = str(REPO_ROOT / "episodes" / "transitions.jsonl")
    ckpt_path = str(REPO_ROOT / "checkpoints" / "local_jepa.npz")
    hash_flag = " --force-hash" if args.force_hash else ""
    hermes_flag = f" --hermes-home {shlex.quote(str(args.hermes_home))}" if args.hermes_home else ""

    payload = {
        "schema": "steppewatch-self-tap.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steppewatch_root": str(REPO_ROOT),
        "target_repo": repo_str,
        "read_only_on_transcripts": True,
        "zero_api_tokens": True,
        "hardware": hw,
        "embedder": embedder.backend_name,
        "self_check": self_check,
        "ruleset_path": str(REPO_ROOT / "rules" / "frontier-agent-watchdog-skill.md"),
        "discovered_transcripts_count": len(transcripts),
        "discovered_transcripts": transcripts,
        "next_commands": {
            "benchmark_local_silicon": f"python3 {REPO_ROOT / 'scripts' / '01_see_the_math_m2.py'}{hash_flag} --repo {repo_str}",
            "watch_subagents_once_json": f"python3 {REPO_ROOT / 'scripts' / '02_watch_frontier_agents.py'}{hash_flag}{hermes_flag} --once --json --episodes {episodes_path}",
            "train_jepa_demo": f"python3 {REPO_ROOT / 'scripts' / '03_train_tiny_jepa.py'}{hash_flag} --demo-synthetic --out {ckpt_path}",
        },
    }
    print(json.dumps(payload, indent=2))
    return 0 if self_check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
