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

from local_embedder import (  # noqa: E402
    LocalEmbedder,
    normalize_step,
    score_pre_embedded_window,
)


def _build_segments(num_steps: int, segment_breaks: List[int]) -> List[Tuple[int, int]]:
    """Convert segment boundary indices into contiguous [start, end) slices."""
    if num_steps <= 0:
        return []
    cuts = sorted({0, num_steps, *(b for b in segment_breaks if 0 < b < num_steps)})
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def replay_session_windows(
    embedder: LocalEmbedder,
    steps: List[str],
    trace_id: str,
    stride: int = 2,
    segment_breaks: List[int] | None = None,
    intent_text: str = "",
) -> Dict[str, Any]:
    """Single-pass batched W=5 replay respecting user-request segment boundaries."""
    capped = steps[-120:]
    offset = len(steps) - len(capped)
    norm_steps = [normalize_step(s) for s in capped]
    end_summary = " | ".join(capped[-2:])[:300] if capped else intent_text[:300]
    drift_intent = (intent_text or "")[:400]

    # Single batched encode for all steps + intent + end_summary (eliminates ~50x per-window calls).
    batch_texts = list(norm_steps)
    intent_idx = len(batch_texts)
    batch_texts.append(drift_intent)
    end_idx = len(batch_texts)
    batch_texts.append(end_summary)
    all_embs = embedder.encode(batch_texts)

    drift_delta = round(
        float(1.0 - np.clip(float(np.dot(all_embs[intent_idx], all_embs[end_idx])), -1.0, 1.0)),
        4,
    )

    adj_breaks = [b - offset for b in (segment_breaks or []) if offset < b < len(steps)]
    segments = _build_segments(len(norm_steps), adj_breaks)

    max_score = 0.0
    first_flag_idx = None
    flagged_preview = None
    flagged_action = "continue"
    wait_chatter_seen = False
    windows_checked = 0

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < 3:
            continue
        if seg_len == 3:
            win_ends = [seg_end]
        else:
            win_ends = list(range(seg_start + 4, seg_end + 1, max(1, stride)))
            if win_ends[-1] != seg_end:
                win_ends.append(seg_end)

        for end_idx_w in win_ends:
            start_idx_w = max(seg_start, end_idx_w - 5)
            res = score_pre_embedded_window(
                norm_steps[start_idx_w:end_idx_w],
                all_embs[start_idx_w:end_idx_w],
                embedder.backend_name,
                trace_id=trace_id,
            )
            windows_checked += 1
            if res.get("wait_chatter_detected"):
                wait_chatter_seen = True
            sc = float(res["thrash_score"])
            if sc > max_score:
                max_score = sc
            if res["thrash_detected"] and first_flag_idx is None:
                first_flag_idx = end_idx_w
                flagged_preview = capped[end_idx_w - 1][:110]
                flagged_action = str(res["action"])

    post_steps = 0
    post_est_tokens = 0
    if first_flag_idx is not None and first_flag_idx < len(capped):
        tail = capped[first_flag_idx:]
        post_steps = len(tail)
        post_est_tokens = sum(max(1, len(s) // 4) for s in tail)

    return {
        "max_thrash_score": round(max_score, 4),
        "flagged": first_flag_idx is not None,
        "flagged_action": flagged_action,
        "flagged_step_preview": flagged_preview,
        "first_flag_step": first_flag_idx,
        "post_flag_steps": post_steps,
        "post_flag_est_tokens": post_est_tokens,
        "windows_checked": windows_checked,
        "intent_drift_delta": drift_delta,
        "wait_chatter_detected": wait_chatter_seen,
        "advisory_only": True,
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
    parser.add_argument("--offline", action="store_true", help="Load local models from cache only; forbid Hugging Face network calls")
    args = parser.parse_args()

    t0 = time.perf_counter()
    watcher = import_module("02_watch_frontier_agents")
    embedder = LocalEmbedder(force_hash=args.force_hash, offline=args.offline)
    targets = watcher.discover_recent_transcripts(max_files=args.max_sessions, hermes_home=args.hermes_home)
    db_preloaded = watcher.preload_database_sessions(targets, max_lines=600)

    rows: List[Dict[str, Any]] = []
    exported: List[Dict[str, Any]] = []
    total_post_steps = 0
    total_post_tokens = 0
    total_acks_filtered = 0
    total_wrappers_skipped = 0
    placeholder_intents_count = 0

    for source, path in targets:
        pre = db_preloaded.get(path) if isinstance(path, watcher.DatabaseSession) else None
        parsed = watcher.parse_jsonl_transcript(path, max_lines=600, preloaded_records=pre)
        steps = parsed["steps"]
        if len(steps) < 3:
            continue

        # Measure drift against the active request intent when multi-request turns exist.
        active_intent = parsed.get("latest_intent") or parsed["intent"]
        replay = replay_session_windows(
            embedder,
            steps,
            trace_id=path.stem,
            segment_breaks=parsed.get("segment_breaks"),
            intent_text=active_intent,
        )
        total_acks_filtered += int(parsed.get("filtered_acks", 0))
        total_wrappers_skipped += int(parsed.get("compaction_wrappers_skipped", 0))
        if parsed.get("intent_is_placeholder"):
            placeholder_intents_count += 1
        total_post_steps += replay["post_flag_steps"]
        total_post_tokens += replay["post_flag_est_tokens"]

        curr_txt = steps[0][:300]
        intent_txt = active_intent[:400]
        end_txt = " | ".join(steps[-2:])[:300]
        drift_delta = replay["intent_drift_delta"]

        row = {
            "source": source,
            "session": path.name,
            "steps": len(steps),
            "acks_filtered": int(parsed.get("filtered_acks", 0)),
            "flagged": replay["flagged"],
            "advisory_only": True,
            "flagged_action": replay["flagged_action"],
            "wait_chatter_detected": replay["wait_chatter_detected"],
            "max_thrash_score": replay["max_thrash_score"],
            "first_flag_step": replay["first_flag_step"],
            "post_flag_steps": replay["post_flag_steps"],
            "post_flag_est_tokens": replay["post_flag_est_tokens"],
            "intent_drift_delta": drift_delta,
            "intent_is_placeholder": bool(parsed.get("intent_is_placeholder", False)),
            "touched_files_count": len(parsed["touched_files"]),
            "intent_preview": intent_txt[:110],
            "flagged_step_preview": replay["flagged_step_preview"],
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
                    "intended_outcome": {
                        "summary": intent_txt,
                        "is_placeholder": bool(parsed.get("intent_is_placeholder", False)),
                    },
                    "local_prediction": {"source": "zero_shot_intent_prior", "predicted_stall_risk": 0.2},
                    "end_state": {
                        "summary": end_txt,
                        "touched_files": parsed["touched_files"][:12],
                        "thrash_detected": replay["flagged"],
                        "thrash_score": replay["max_thrash_score"],
                        "action": replay["flagged_action"],
                        "advisory_only": True,
                    },
                    "assessment_vs_expectations": {
                        "state_surprise_delta": drift_delta,
                        "risk_prediction_error": round(abs(actual_risk - 0.2), 4),
                        "actual_risk_target": round(actual_risk, 4),
                        "target_source": "heuristic_detector_proxy",
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
        "embedder_fallback_reason": embedder.fallback_reason,
        "advisory_only": True,
        "elapsed_ms": elapsed_ms,
        "sessions_scanned": len(rows),
        "sessions_by_source": {source: sum(r["source"] == source for r in rows) for source in sorted({r["source"] for r in rows})},
        "sessions_flagged": len(flagged_rows),
        "bare_tool_acks_filtered": total_acks_filtered,
        "compaction_wrappers_skipped": total_wrappers_skipped,
        "placeholder_intents_count": placeholder_intents_count,
        "estimated_post_stall_steps": total_post_steps,
        "estimated_post_stall_tokens": total_post_tokens,
        "post_flag_est_tokens_note": "Character-derived step-summary count (chars/4) including advisory/unverified flags; not API billing or proven savings.",
        "exported_episodes_count": len(exported),
        "exported_episodes_path": str(args.export_episodes) if args.export_episodes else None,
        "top_sessions": rows_sorted[:10],
    }

    if args.json_mode:
        print(json.dumps(report, indent=2))
        return 0

    print(f"=== Steppewatch Session X-Ray ({len(rows)} real local transcripts in {elapsed_ms} ms, $0.00 API cost) ===")
    print(f"Embedder                  : {embedder.backend_name}")
    if embedder.fallback_reason:
        print(f"[note] {embedder.fallback_reason}")
    if not rows:
        print(
            "No local transcripts with >= 3 substantive steps found under "
            "~/.claude/projects, ~/.codex/sessions, or ~/.hermes/{sessions,logs/sessions,profiles/*/sessions,state.db}."
        )
        print("Try passing --hermes-home <path> or test an explicit file with `python3 scripts/02_watch_frontier_agents.py --trace <file> --once`.")
        return 0
    print(f"Advisory loop flags       : {len(flagged_rows)} / {len(rows)} (advisory only — inspect before intervening)")
    print(f"Bare tool-acks filtered   : {total_acks_filtered} | compaction wrappers skipped: {total_wrappers_skipped}")
    print(f"Post-flag steps observed  : {total_post_steps} (~{total_post_tokens} char/4 step-summary units; not billing/savings)")
    if rows_sorted:
        print("\nTop sessions by advisory loop score / intent-drift surprise:")
        for r in rows_sorted[:8]:
            tag = "ADVISORY" if r["flagged"] else "OK      "
            print(
                f"  [{tag}] ({r['source']:15s}) {r['session'][:34]:34s} "
                f"| steps={r['steps']:3d} max_thrash={r['max_thrash_score']:.3f} "
                f"drift={r['intent_drift_delta']:.3f} post_steps={r['post_flag_steps']}"
            )
            if r["flagged"]:
                print(f"            intent : {r['intent_preview']}")
                if r.get("flagged_step_preview"):
                    print(f"            flag@{r['first_flag_step']}: {r['flagged_step_preview']}")
    if args.export_episodes is not None:
        print(f"\nExported {len(exported)} real 5-field JEPA episodes -> {args.export_episodes}")
        if len(exported) >= 6:
            print("Train your own personal JEPA watchdog on YOUR real history right now:")
            print(f"  python3 scripts/03_train_tiny_jepa.py --episodes {args.export_episodes} --out checkpoints/my_real_jepa.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
