#!/usr/bin/env python3
"""Step 2 — Read-Only Live Frontier Agent & Subagent Watcher + JEPA Episode Collector.

Tails local append-only JSONL transcripts already written by:
  - Claude Code: ~/.claude/projects/*/*.jsonl and <session>/subagents/agent-*.jsonl
  - Codex CLI:   ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
  - Hermes:      ~/.hermes/{logs/sessions,sessions,profiles/*/sessions}/*.jsonl
                 plus read-only state.db messages when JSONL is absent.
  - Or any explicit --trace JSONL file.

Zero API keys, zero frontier tokens, read-only on transcripts. Filters terse
bare tool-acknowledgement stubs (preventing false loop alerts during fast
parallel edit bursts), alerts on genuine stalls/loops, and appends 5-field
JEPA Residual Episodes to JSONL.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hermes_transcripts import (
    DatabaseSession,
    Transcript,
    discover_hermes,
    read_database_session,
    read_database_sessions_batch,
)

from local_embedder import (
    LocalEmbedder,
    detect_agent_thrash,
    predict_with_jepa_checkpoint,
)

# Live calibration lesson (Proof Step B): terse parallel tool-ack stubs
# ("Created file." x4 or "Made 1 replacement..." x3) lack semantic targets
# and trigger false-positive thrash scores unless filtered out before windowing.
BARE_TOOL_ACK_RE = re.compile(
    r"^(?:"
    r"created\s+file\.?|"
    r"updated\s+file\.?|"
    r"file\s+(?:created|updated|written|saved)\.?|"
    r"made\s+\d+\s+replacements?(?:\s+in\s+\S+)?\.?|"
    r"ok\.?|done\.?|success\.?|acknowledged\.?|noted\.?|got\s+it\.?|"
    r"command\s+exited\s+with\s+code\s+0\.?"
    r")$",
    re.I,
)

COMPACTION_WRAPPER_RE = re.compile(
    r"^(?:"
    r"<(?:command-name|command-message|local-command-|system-reminder|context-compaction)|"
    r"this\s+session\s+is\s+being\s+continued\s+from\s+a\s+previous\s+conversation|"
    r"\[?compaction\]?\s*[:\-]|"
    r"conversation\s+compacted|"
    r"summary\s+of\s+previous\s+conversation"
    r")",
    re.I,
)


def is_bare_tool_ack(text: str) -> bool:
    """Return True if a trace line is a terse framework ack stub without substance."""
    compact = " ".join(text.split()).strip()
    if len(compact) < 12 and compact.lower() in {"ok", "ok.", "done", "done.", "true", "null"}:
        return True
    return bool(BARE_TOOL_ACK_RE.match(compact))


def is_compaction_or_wrapper_text(text: str) -> bool:
    """Return True if text is a harness compaction or slash-command wrapper rather than real intent."""
    compact = " ".join((text or "").split()).strip()
    if not compact:
        return True
    return bool(COMPACTION_WRAPPER_RE.match(compact))


def _read_tail_lines(path: Path, max_lines: int, max_tail_bytes: int = 2 * 1024 * 1024) -> Tuple[List[str], int]:
    """Read the last `max_lines` of a JSONL file without loading 500+ MB logs into RAM."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_tail_bytes:
                f.seek(size - max_tail_bytes)
                raw_bytes = f.read()
                lines = raw_bytes.decode("utf-8", errors="ignore").splitlines()
                # Discard partial first line after mid-file seek
                if lines:
                    lines = lines[1:]
                return lines[-max_lines:], len(lines)
            raw_bytes = f.read()
        lines = raw_bytes.decode("utf-8", errors="ignore").splitlines()
        return lines[-max_lines:], len(lines)
    except OSError:
        return [], 0


def _read_head_intent(path: Path, max_head_bytes: int = 64 * 1024) -> str:
    """Scan the top of a JSONL transcript for the opening substantive user intent."""
    try:
        with path.open("rb") as f:
            raw_bytes = f.read(max_head_bytes)
    except OSError:
        return ""
    for raw in raw_bytes.decode("utf-8", errors="ignore").splitlines()[:80]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        rec_type = str(obj.get("type") or obj.get("role") or "").lower()
        if rec_type in ("session_meta", "system"):
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        msg_obj = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        is_user = rec_type in ("user", "user_input") or str(payload.get("type", "")).lower() == "user_message"
        if not is_user:
            continue
        text = (
            _extract_text_from_content(msg_obj.get("content"))
            or _extract_text_from_content(payload.get("message"))
            or _extract_text_from_content(payload.get("content"))
            or _extract_text_from_content(obj.get("content"))
            or _extract_text_from_content(obj.get("text"))
        ).strip()
        compact = " ".join(text.split())
        if len(compact) >= 8 and not is_compaction_or_wrapper_text(compact):
            return compact[:400]
    return ""


def discover_recent_transcripts(max_files: int = 8, hermes_home: Optional[Path] = None) -> List[Tuple[str, Transcript]]:
    """Find the most recently modified Claude Code, Codex, and Hermes JSONL logs."""
    if max_files <= 0:
        return []
    home = Path.home()
    patterns = [
        ("claude_main", str(home / ".claude" / "projects" / "*" / "*.jsonl")),
        ("claude_subagent", str(home / ".claude" / "projects" / "*" / "*" / "subagents" / "agent-*.jsonl")),
        ("codex", str(home / ".codex" / "sessions" / "*" / "*" / "*" / "rollout-*.jsonl")),
    ]
    found: List[Tuple[float, str, Transcript]] = []
    for source, pat in patterns:
        for path_str in glob.glob(pat):
            p = Path(path_str)
            try:
                found.append((p.stat().st_mtime, source, p))
            except OSError:
                continue
    found.extend(discover_hermes((hermes_home or home / ".hermes").expanduser(), max_files))
    found.sort(key=lambda item: item[0], reverse=True)
    return [(src, p) for _, src, p in found[:max_files]]


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if isinstance(t, str):
                    parts.append(t)
                elif item.get("type") == "tool_use":
                    name = item.get("name", "tool")
                    inp = json.dumps(item.get("input", {}))[:160]
                    parts.append(f"tool:{name}({inp})")
        return " ".join(parts)
    if isinstance(content, dict):
        return _extract_text_from_content(content.get("content") or content.get("text") or "")
    return ""


def parse_jsonl_transcript(
    path: Transcript,
    max_lines: int = 400,
    max_tail_bytes: int = 2 * 1024 * 1024,
    preloaded_records: Optional[Tuple[List[Dict[str, Any]], int]] = None,
) -> Dict[str, Any]:
    """Extract normalized intent, substantive agent steps, request boundaries, and file hints."""
    if preloaded_records is not None:
        tail, line_count = preloaded_records
        head_intent = ""
    elif isinstance(path, DatabaseSession):
        tail, line_count = read_database_session(path, max_lines)
        head_intent = ""
    else:
        head_intent = _read_head_intent(path)
        tail, line_count = _read_tail_lines(path, max_lines=max_lines, max_tail_bytes=max_tail_bytes)

    intent = head_intent
    latest_intent = head_intent
    steps: List[str] = []
    segment_breaks: List[int] = []
    touched: List[str] = []
    filtered_acks = 0
    compaction_wrappers_skipped = 0
    file_re = re.compile(r"\b([A-Za-z0-9_.\-/]+\.(?:py|cs|ts|tsx|js|go|rs|json|jsonl|yaml|yml|md|sh|txt|log))\b")

    for raw in tail:
        try:
            obj = raw if isinstance(raw, dict) else json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue

        rec_type = str(obj.get("type") or obj.get("role") or "").lower()
        if rec_type in ("session_meta", "system"):
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        msg_obj = obj.get("message") if isinstance(obj.get("message"), dict) else {}

        text = (
            _extract_text_from_content(msg_obj.get("content"))
            or _extract_text_from_content(payload.get("message"))
            or _extract_text_from_content(payload.get("content"))
            or _extract_text_from_content(obj.get("content"))
            or _extract_text_from_content(obj.get("text"))
        ).strip()

        calls = obj.get("tool_calls") or msg_obj.get("tool_calls") or []
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", call)
                if isinstance(function, dict) and function.get("name"):
                    text += f" tool:{function['name']}({str(function.get('arguments', ''))[:160]})"
        text = text.strip()

        if not text:
            tool_name = obj.get("tool_name") or obj.get("tool") or payload.get("name") or msg_obj.get("name")
            if tool_name:
                text = f"tool:{tool_name} {json.dumps(payload or obj)[:180]}"

        if not text:
            continue

        is_user_only = rec_type in ("user", "user_input") or str(payload.get("type", "")).lower() == "user_message"
        compact_full = " ".join(text.split())

        if is_compaction_or_wrapper_text(compact_full):
            compaction_wrappers_skipped += 1
            if steps and (not segment_breaks or segment_breaks[-1] != len(steps)):
                segment_breaks.append(len(steps))
            continue

        if is_user_only:
            if len(compact_full) >= 8:
                if not intent:
                    intent = compact_full[:400]
                latest_intent = compact_full[:400]
                if steps and (not segment_breaks or segment_breaks[-1] != len(steps)):
                    segment_breaks.append(len(steps))
            continue

        if not intent and "INTENT:" in compact_full:
            intent = compact_full[:400]
            latest_intent = intent

        compact = compact_full[:500]
        if is_bare_tool_ack(compact):
            filtered_acks += 1
            continue
        steps.append(compact)
        for fn in file_re.findall(compact):
            if fn not in touched:
                touched.append(fn)

    intent_is_placeholder = not bool(intent)
    resolved_intent = intent or f"Supervise session {path.name}"
    return {
        "intent": resolved_intent,
        "latest_intent": latest_intent or resolved_intent,
        "intent_is_placeholder": intent_is_placeholder,
        "steps": steps,
        "segment_breaks": segment_breaks,
        "touched_files": touched[:25],
        "line_count": line_count,
        "filtered_acks": filtered_acks,
        "compaction_wrappers_skipped": compaction_wrappers_skipped,
    }


def build_residual_episode(
    embedder: LocalEmbedder,
    source: str,
    trace_path: Transcript,
    parsed: Dict[str, Any],
    thrash: Dict[str, Any],
    explicit_intent: Optional[str] = None,
    jepa_ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct one 5-field Prediction-Residual JEPA Episode."""
    steps = parsed["steps"]
    current_summary = steps[0] if steps else f"Session start ({source}: {trace_path.name})"
    end_summary = " | ".join(steps[-2:]) if steps else current_summary
    intent_text = (explicit_intent or parsed.get("latest_intent") or parsed["intent"])[:400]

    embs = embedder.encode([current_summary, intent_text, end_summary])
    z_curr, z_intent, z_end = embs[0], embs[1], embs[2]

    pred = predict_with_jepa_checkpoint(jepa_ckpt or "", z_curr, z_intent)
    if pred is not None:
        z_hat = pred["predicted_end_emb"]
        pred_risk = float(pred["predicted_stall_risk"])
        pred_source = f"local_jepa:{Path(pred['checkpoint']).name}"
    else:
        z_hat = z_intent
        pred_risk = 0.20
        pred_source = "zero_shot_intent_prior"

    state_surprise = round(float(1.0 - np.clip(float(np.dot(z_hat, z_end)), -1.0, 1.0)), 4)
    actual_risk = 1.0 if thrash.get("thrash_detected") else float(thrash.get("thrash_score", 0.0))
    risk_error = round(abs(actual_risk - pred_risk), 4)

    return {
        "schema": "jepa-residual-episode.v1",
        "episode_id": f"ep-{int(time.time())}-{trace_path.stem[:16]}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_agent": source,
        "trace_path": str(trace_path),
        "current_state": {
            "summary": current_summary[:300],
            "step_count_observed": len(steps),
            "filtered_tool_acks": int(parsed.get("filtered_acks", 0)),
        },
        "intended_outcome": {
            "summary": intent_text,
            "is_placeholder": bool(parsed.get("intent_is_placeholder", False)) if not explicit_intent else False,
        },
        "local_prediction": {
            "source": pred_source,
            "predicted_stall_risk": pred_risk,
        },
        "end_state": {
            "summary": end_summary[:300],
            "touched_files": parsed["touched_files"][:12],
            "thrash_detected": bool(thrash.get("thrash_detected")),
            "thrash_score": float(thrash.get("thrash_score", 0.0)),
            "action": thrash.get("action", "continue"),
            "advisory_only": True,
        },
        "assessment_vs_expectations": {
            "state_surprise_delta": state_surprise,
            "risk_prediction_error": risk_error,
            "actual_risk_target": round(actual_risk, 4),
            "target_source": "heuristic_detector_proxy",
            "human_label": None,
        },
    }


def preload_database_sessions(
    targets: List[Tuple[str, Transcript]],
    max_lines: int,
) -> Dict[DatabaseSession, Tuple[List[Dict[str, Any]], int]]:
    """Batch-read all DatabaseSession targets grouped by state.db (1 capture per DB)."""
    grouped: Dict[Path, List[DatabaseSession]] = defaultdict(list)
    for _, target in targets:
        if isinstance(target, DatabaseSession):
            grouped[target.database].append(target)
    preloaded: Dict[DatabaseSession, Tuple[List[Dict[str, Any]], int]] = {}
    for db_path, sessions in grouped.items():
        batch = read_database_sessions_batch(db_path, [s.session_id for s in sessions], max_lines)
        for sess in sessions:
            preloaded[sess] = batch.get(sess.session_id, ([], 0))
    return preloaded


def evaluate_once(
    embedder: LocalEmbedder,
    targets: List[Tuple[str, Transcript]],
    episodes_path: Path,
    explicit_intent: Optional[str],
    jepa_ckpt: Optional[str],
    last_seen_lines: Dict[str, int],
    json_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate all target transcripts once and append new residual episodes."""
    results: List[Dict[str, Any]] = []
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    db_preloaded = preload_database_sessions(targets, max_lines=400)
    for source, path in targets:
        pre = db_preloaded.get(path) if isinstance(path, DatabaseSession) else None
        parsed = parse_jsonl_transcript(path, preloaded_records=pre)
        key = str(path)
        if parsed["line_count"] == last_seen_lines.get(key):
            continue
        last_seen_lines[key] = parsed["line_count"]

        # Only score steps within the active request segment so prior turns don't bleed into tail checks.
        steps = parsed["steps"]
        breaks = parsed.get("segment_breaks") or []
        active_steps = steps[breaks[-1]:] if breaks and breaks[-1] < len(steps) else steps
        thrash = detect_agent_thrash(embedder, active_steps, trace_id=path.stem)
        ep = build_residual_episode(
            embedder=embedder,
            source=source,
            trace_path=path,
            parsed=parsed,
            thrash=thrash,
            explicit_intent=explicit_intent,
            jepa_ckpt=jepa_ckpt,
        )
        with episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep) + "\n")

        delta = ep["assessment_vs_expectations"]["state_surprise_delta"]
        pred_r = ep["local_prediction"]["predicted_stall_risk"]
        summary_row = {
            "source": source,
            "trace_path": str(path),
            "steps_observed": len(parsed["steps"]),
            "filtered_tool_acks": parsed["filtered_acks"],
            "thrash_detected": bool(thrash["thrash_detected"]),
            "thrash_score": float(thrash["thrash_score"]),
            "action": thrash["action"],
            "advisory_only": True,
            "wait_chatter_detected": bool(thrash.get("wait_chatter_detected", False)),
            "predicted_stall_risk": pred_r,
            "state_surprise_delta": delta,
            "latency_ms": thrash["latency_ms"],
            "episode_id": ep["episode_id"],
        }
        results.append(summary_row)

        if not json_mode:
            flag = f"ADVISORY:{thrash['action'].upper()}" if thrash["thrash_detected"] else "OK"
            print(
                f"[{time.strftime('%H:%M:%S')}] [{flag}] ({source}) {path.name} "
                f"| steps={len(parsed['steps'])} (acks_skipped={parsed['filtered_acks']}) "
                f"thrash={thrash['thrash_score']:.3f} pred_risk={pred_r:.2f} "
                f"surprise_delta={delta:.3f} ({thrash['latency_ms']} ms)"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only local watcher for Claude Code, Codex CLI, and Hermes transcripts."
    )
    parser.add_argument("--hermes-home", type=Path, default=None, help="Hermes root containing sessions, state.db and profiles (default: ~/.hermes)")
    parser.add_argument("--trace", type=Path, help="Explicit JSONL transcript file to watch")
    parser.add_argument("--intent", type=str, default=None, help="Optional explicit intended_outcome override")
    parser.add_argument("--episodes", type=Path, default=Path("episodes/transitions.jsonl"), help="Output NDJSON episodes file")
    parser.add_argument("--jepa-checkpoint", type=str, default=None, help="Optional trained .npz local JEPA checkpoint")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval in continuous watch mode")
    parser.add_argument("--max-sessions", type=int, default=6, help="Max auto-discovered sessions to watch")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Emit machine-readable JSON output for agent self-tap")
    parser.add_argument("--force-hash", action="store_true", help="Use zero-download hash384 embedder")
    parser.add_argument("--offline", action="store_true", help="Load local models from cache only; forbid Hugging Face network calls")
    args = parser.parse_args()

    if args.trace is not None:
        trace_path = args.trace.expanduser()
        if not trace_path.is_file():
            msg = f"Explicit --trace file not found: {args.trace}"
            if args.json_mode:
                print(json.dumps({"schema": "steppewatch-watch-report.v1", "error": msg, "results": []}, indent=2))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            return 1
        args.trace = trace_path

    embedder = LocalEmbedder(force_hash=args.force_hash, offline=args.offline)
    if not args.json_mode:
        print(f"Steppewatch active | embedder={embedder.backend_name} | episodes={args.episodes}")
        if embedder.fallback_reason:
            print(f"[note] {embedder.fallback_reason}")

    last_seen_lines: Dict[str, int] = {}
    while True:
        targets = [("explicit", args.trace)] if args.trace else discover_recent_transcripts(args.max_sessions, hermes_home=args.hermes_home)
        if not targets:
            if args.json_mode:
                print(json.dumps({
                    "schema": "steppewatch-watch-report.v1",
                    "embedder": embedder.backend_name,
                    "episodes_path": str(args.episodes),
                    "alerts_count": 0,
                    "results": [],
                }, indent=2))
            else:
                print("No local Claude Code / Codex / Hermes JSONL transcripts found yet (pass --trace to test a file).")
            if args.once:
                return 0
        else:
            results = evaluate_once(
                embedder=embedder,
                targets=targets,
                episodes_path=args.episodes,
                explicit_intent=args.intent,
                jepa_ckpt=args.jepa_checkpoint,
                last_seen_lines=last_seen_lines,
                json_mode=args.json_mode,
            )
            if args.json_mode:
                print(json.dumps({
                    "schema": "steppewatch-watch-report.v1",
                    "embedder": embedder.backend_name,
                    "episodes_path": str(args.episodes),
                    "alerts_count": sum(1 for r in results if r["thrash_detected"]),
                    "results": results,
                }, indent=2))
            if args.once:
                return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
