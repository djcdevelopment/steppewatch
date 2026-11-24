#!/usr/bin/env python3
"""Step 4 — Retrospective X-Ray of Your Own Real Frontier Agent Transcripts.

100% read-only, zero API tokens, zero installs beyond numpy.
Scans your existing local Claude Code (+ subagents), Codex CLI, and Hermes JSONL
logs on disk to:
  1. Replay sliding windows (W=5) across real past sessions and find the exact
     turn where Steppewatch would have flagged silent stall/thrash loops.
  2. Estimate wasted post-stall turns & characters burned after the catch point.
  3. Rank sessions by intent-to-outcome drift (surprise delta).
  4. Optionally export real 5-field JEPA Residual Episodes directly from your
     existing transcript archive so you can train `03_train_tiny_jepa.py` on
     your own real data immediately — without waiting weeks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from local_embedder import LocalEmbedder, detect_agent_thrash  # noqa: E402


def replay_session_windows(
    embedder: LocalEmbedder,
    steps: List[str],
    trace_id: str,
    stride: int = 2,
) -> Dict[str, Any]:
    """Replay W=5 sliding windows across a session to find peak thrash & first catch."""
    if len(steps) < 4:
        res = detect_agent_thrash(embedder, steps, trace_id=trace_id)
        return {
            "max_thrash_score": float(res["thrash_score"]),
            "flagged": bool(res["thrash_detected"]),
            "first_flag_step": len(steps) if res["thrash_detected"] else None,
            "post_flag_steps": 0,
            "post_flag_est_tokens": 0,
            "windows_checked": 1,
        }

    # Cap total steps scanned per giant session so X-Ray stays fast (<5 s total)
    capped = steps[-120:]
    max_score = 0.0
    first_flag_idx = None
    windows_checked = 0

    for end_idx in range(4, len(capped) + 1, max(1, stride)):
        win = capped[max(0, end_idx - 5) : end_idx]
        res = detect_agent_thrash(embedder, win, trace_id=trace_id)
        windows_checked += 1
        sc = float(res["thrash_score"])
        if sc > max_score:
            max_score = sc
        if res["thrash_detected"] and first_flag_idx is None:
            first_flag_idx = end_idx

    post_steps = 0
    post_est_tokens = 0
    if first_flag_idx is not None and first_flag_idx < len(capped):
        tail = capped[first_flag_idx:]
        post_steps = len(tail)
        post_est_tokens = sum(max(1, len(s) // 4) for s in tail)

    return {
        "max_thrash_score": round(max_score, 4),
        "flagged": first_flag_idx is not None,
        "first_flag_step": first_flag_idx,
        "post_flag_steps": post_steps,
        "post_flag_est_tokens": post_est_tokens,
        "windows_checked": windows_checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only retrospective X-Ray of your real Claude Code, Codex, and Hermes sessions."
    )
    parser.add_argument("--hermes-home", type=Path, default=None, help="Hermes root containing sessions, state.db and profiles (default: ~/.hermes)")
    parser.add_argument("--max-sessions", type=int, default=20, help="Max recent local transcripts to scan")
    parser.add_argument("--export-episodes", type=Path, default=None, help="Optional NDJSON path to seed real JEPA training episodes")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Emit structured JSON output")
    parser.add_argument("--force-hash", action="store_true", help="Use zero-download hash384 embedder")
    args = parser.parse_args()

    t0 = time.perf_counter()
    watcher = import_module("02_watch_frontier_agents")
    embedder = LocalEmbedder(force_hash=args.force_hash)
    targets = watcher.discover_recent_transcripts(max_files=args.max_sessions, hermes_home=args.hermes_home)

    rows: List[Dict[str, Any]] = []
    exported: List[Dict[str, Any]] = []
    total_post_steps = 0
    total_post_tokens = 0
    total_acks_filtered = 0

    for source, path in targets:
        parsed = watcher.parse_jsonl_transcript(path, max_lines=600)
        steps = parsed["steps"]
        if len(steps) < 3:
            continue

        replay = replay_session_windows(embedder, steps, trace_id=path.stem)
        total_acks_filtered += int(parsed.get("filtered_acks", 0))
        total_post_steps += replay["post_flag_steps"]
        total_post_tokens += replay["post_flag_est_tokens"]

        # Compute intent-to-end-state drift (surprise delta)
        curr_txt = steps[0][:300]
        intent_txt = parsed["intent"][:400]
        end_txt = " | ".join(steps[-2:])[:300]
        embs = embedder.encode([curr_txt, intent_txt, end_txt])
        drift_delta = round(float(1.0 - np.clip(float(np.dot(embs[1], embs[2])), -1.0, 1.0)), 4)

        row = {
            "source": source,
            "session": path.name,
            "steps": len(steps),
            "acks_filtered": int(parsed.get("filtered_acks", 0)),
            "flagged": replay["flagged"],
            "max_thrash_score": replay["max_thrash_score"],
            "first_flag_step": replay["first_flag_step"],
            "post_flag_steps": replay["post_flag_steps"],
            "post_flag_est_tokens": replay["post_flag_est_tokens"],
            "intent_drift_delta": drift_delta,
            "touched_files_count": len(parsed["touched_files"]),
        }
        rows.append(row)

        if args.export_episodes is not None:
            actual_risk = 1.0 if replay["flagged"] else replay["max_thrash_score"]
            exported.append(
                {
                    "schema": "jepa-residual-episode.v1",
                    "episode_id": f"xray-{path.stem[:20]}",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_agent": source,
                    "current_state": {"summary": curr_txt, "step_count_observed": len(steps)},
                    "intended_outcome": {"summary": intent_txt},
                    "local_prediction": {"source": "zero_shot_intent_prior", "predicted_stall_risk": 0.2},
                    "end_state": {
                        "summary": end_txt,
                        "touched_files": parsed["touched_files"][:12],
                        "thrash_detected": replay["flagged"],
                        "thrash_score": replay["max_thrash_score"],
                        "action": "early_abort_and_reroute" if replay["flagged"] else "continue",
                    },
                    "assessment_vs_expectations": {
                        "state_surprise_delta": drift_delta,
                        "risk_prediction_error": round(abs(actual_risk - 0.2), 4),
                        "actual_risk_target": round(actual_risk, 4),
                        "human_label": None,
                    },
                }
            )

    if args.export_episodes is not None and exported:
        args.export_episodes.parent.mkdir(parents=True, exist_ok=True)
        with args.export_episodes.open("w", encoding="utf-8") as f:
            for ep in exported:
                f.write(json.dumps(ep) + "\n")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    flagged_rows = [r for r in rows if r["flagged"]]
    rows_sorted = sorted(rows, key=lambda r: (r["flagged"], r["max_thrash_score"], r["intent_drift_delta"]), reverse=True)

    report = {
        "schema": "steppewatch-session-xray.v1",
        "embedder": embedder.backend_name,
        "elapsed_ms": elapsed_ms,
        "sessions_scanned": len(rows),
        "sessions_by_source": {source: sum(r["source"] == source for r in rows) for source in sorted({r["source"] for r in rows})},
        "sessions_flagged": len(flagged_rows),
        "bare_tool_acks_filtered": total_acks_filtered,
        "estimated_post_stall_steps": total_post_steps,
        "estimated_post_stall_tokens": total_post_tokens,
        "exported_episodes_count": len(exported),
        "exported_episodes_path": str(args.export_episodes) if args.export_episodes else None,
        "top_sessions": rows_sorted[:10],
    }

    if args.json_mode:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== Steppewatch Session X-Ray ({len(rows)} real local transcripts in {elapsed_ms} ms, $0.00 API cost) ===")
    print(f"Embedder                  : {embedder.backend_name}")
    print(f"Flagged stall/loop runs   : {len(flagged_rows)} / {len(rows)}")
    print(f"Bare tool-acks filtered   : {total_acks_filtered}")
    print(f"Post-flag steps observed  : {total_post_steps} (~{total_post_tokens} est. step-summary tokens)")
    if rows_sorted:
        print("\nTop sessions by thrash / intent-drift surprise:")
        for r in rows_sorted[:8]:
            tag = "FLAGGED" if r["flagged"] else "OK     "
            print(
                f"  [{tag}] ({r['source']:15s}) {r['session'][:34]:34s} "
                f"| steps={r['steps']:3d} max_thrash={r['max_thrash_score']:.3f} "
                f"drift={r['intent_drift_delta']:.3f} post_steps={r['post_flag_steps']}"
            )
    if args.export_episodes is not None:
        print(f"\nExported {len(exported)} real 5-field JEPA episodes -> {args.export_episodes}")
        if len(exported) >= 6:
            print("Train your own personal JEPA watchdog on YOUR real history right now:")
            print(f"  python3 scripts/03_train_tiny_jepa.py --episodes {args.export_episodes} --out checkpoints/my_real_jepa.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
