"""Offline regression tests covering Apple M2 Option B feedback (Issue #2) and PR #1 edge cases."""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

embedder_mod = importlib.import_module("local_embedder")
watcher = importlib.import_module("02_watch_frontier_agents")
xray = importlib.import_module("04_xray_my_sessions")
math_bench = importlib.import_module("01_see_the_math_m2")
jepa_mod = importlib.import_module("03_train_tiny_jepa")
hermes_adapter = importlib.import_module("hermes_transcripts")


class Issue2CalibrationAndParityTest(unittest.TestCase):
    def setUp(self):
        self.embedder = embedder_mod.LocalEmbedder(force_hash=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_issue2_rank1_true_positive_cli_lock_loop_flags_as_advisory_error_loop(self):
        """Issue #2 Rank #1 TP: repeated CLI attempts hitting the same update/lock failure."""
        steps = [
            "Step 1: Running hermes update via CLI; failed with database is locked error rc=1.",
            "Step 2: Retrying hermes update command; failed again with database is locked error rc=1.",
            "Step 3: Running hermes update once more; still failed with database is locked error rc=1.",
            "Step 4: Attempting hermes update again; failed with database is locked error rc=1.",
            "Step 5: Retrying hermes update CLI; cannot acquire lock error rc=1.",
        ]
        res = embedder_mod.detect_agent_thrash(self.embedder, steps, trace_id="tp-cli-lock")
        self.assertTrue(res["thrash_detected"], f"Expected CLI lock loop to flag, got {res}")
        self.assertEqual(res["action"], "advisory_inspect_error_loop")
        self.assertTrue(res["advisory_only"])

    def test_issue2_rank2_false_positive_multi_request_qa_does_not_flag(self):
        """Issue #2 Rank #2 FP: productive investigation/QA spanning distinct user requests."""
        trace = self.root / "multi_req_qa.jsonl"
        records = [
            {"type": "user", "message": {"content": "Check auth_config.yaml and verify the OAuth callback settings."}},
            {"type": "assistant", "message": {"content": "Inspected auth_config.yaml lines 1-60 for callback URL configuration."}},
            {"type": "assistant", "message": {"content": "Checked oauth_handler.py lines 20-85 and summarized the redirect guard."}},
            {"type": "user", "message": {"content": "Now check billing_config.yaml and confirm webhook signing."}},
            {"type": "assistant", "message": {"content": "Inspected billing_config.yaml lines 1-50 for webhook signing key settings."}},
            {"type": "assistant", "message": {"content": "Verified stripe_verifier.py lines 10-70 and reported that HMAC validation is enabled."}},
            {"type": "user", "message": {"content": "Finally inspect deploy_config.yaml for health check timeouts."}},
            {"type": "assistant", "message": {"content": "Reviewed deploy_config.yaml lines 15-45 and answered with the 30s timeout summary."}},
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        parsed = watcher.parse_jsonl_transcript(trace)
        self.assertEqual(parsed["segment_breaks"], [2, 4])
        replay = xray.replay_session_windows(
            self.embedder,
            parsed["steps"],
            trace_id="fp-multi-req",
            segment_breaks=parsed["segment_breaks"],
            intent_text=parsed["latest_intent"],
        )
        self.assertFalse(replay["flagged"], f"Multi-request QA should not false-flag: {replay}")

    def test_issue2_rank3_false_positive_orchestration_distinct_tasks_does_not_flag(self):
        """Issue #2 Rank #3 FP: similar successful orchestration replies across different tasks."""
        steps = [
            "Dispatched task:sync-101 to profile:alpha and recorded completed status for #101.",
            "Dispatched task:sync-102 to profile:beta and recorded completed status for #102.",
            "Dispatched task:sync-103 to profile:gamma and recorded completed status for #103.",
            "Dispatched task:sync-104 to profile:delta and recorded completed status for #104.",
            "Dispatched task:sync-105 to profile:epsilon and reported all 5 orchestration tasks completed.",
        ]
        res = embedder_mod.detect_agent_thrash(self.embedder, steps, trace_id="fp-orchestration")
        self.assertFalse(res["thrash_detected"], f"Distinct successful orchestration tasks should not flag: {res}")
        self.assertGreaterEqual(res["tail_new_targets"], 4)

    def test_issue2_rank4_false_positive_sequential_chunk_reads_does_not_flag(self):
        """Issue #2 Rank #4 FP: reading sequential transcript chunks and producing a new note."""
        steps = [
            "Read session_transcript.jsonl chunk 1 lines 1-200 to inspect opening discussion.",
            "Read session_transcript.jsonl chunk 2 lines 201-400 to inspect follow-up questions.",
            "Read session_transcript.jsonl chunk 3 lines 401-600 to review benchmark numbers.",
            "Read session_transcript.jsonl chunk 4 lines 601-800 and cross-checked findings_draft.txt.",
            "Produced new note summary_notes.md consolidating all four chunks and verified key citations.",
        ]
        res = embedder_mod.detect_agent_thrash(self.embedder, steps, trace_id="fp-chunk-reads")
        self.assertFalse(res["thrash_detected"], f"Sequential chunk reads should not flag: {res}")
        self.assertGreaterEqual(res["tail_new_targets"], 4)

    def test_issue2_rank5_false_positive_scheduled_deadline_wait_does_not_flag(self):
        """Issue #2 Rank #5 FP: repeated acknowledgments waiting for a requested deadline."""
        steps = [
            "Acknowledged request; waiting for the scheduled deadline window before finalizing the report.",
            "Standing by and still waiting for the scheduled deadline; will check back in 5 minutes.",
            "Standing by and waiting for the scheduled finish timer; no errors observed.",
            "Still waiting for the scheduled deadline completion; will check again at the target time.",
            "Scheduled deadline arrived; standing by to run the final publish step.",
        ]
        res = embedder_mod.detect_agent_thrash(self.embedder, steps, trace_id="fp-scheduled-wait")
        self.assertFalse(res["thrash_detected"], f"Scheduled wait chatter should not flag as hard stall: {res}")
        self.assertTrue(res["wait_chatter_detected"])

    def test_polite_stall_still_flags_as_advisory_and_healthy_stays_zero(self):
        """Synthetic smoke check still separates healthy debugging from polite paraphrased stall."""
        demo = math_bench.run_thrash_demo(self.embedder)
        self.assertFalse(demo["healthy_trace"]["thrash_detected"])
        self.assertEqual(demo["healthy_trace"]["thrash_score"], 0.0)
        self.assertTrue(demo["polite_stall_loop"]["thrash_detected"])
        self.assertEqual(demo["polite_stall_loop"]["action"], "advisory_inspect_loop")
        self.assertTrue(demo["polite_stall_loop"]["advisory_only"])

    def test_single_pass_batched_xray_encodes_once_per_session(self):
        """X-Ray must encode all session steps + intent + end in ONE batched embedder call."""
        steps = [
            f"Step {i + 1}: Let me carefully review the webhook integration requirements again before proposing an edit."
            for i in range(24)
        ]
        call_count = 0
        orig_encode = self.embedder.encode

        def counting_encode(texts):
            nonlocal call_count
            call_count += 1
            return orig_encode(texts)

        with patch.object(self.embedder, "encode", side_effect=counting_encode):
            replay = xray.replay_session_windows(
                self.embedder,
                steps,
                trace_id="batch-test",
                intent_text="Fix webhook retry timeout",
            )
        self.assertEqual(call_count, 1, f"Expected 1 batched encode call, saw {call_count}")
        self.assertTrue(replay["flagged"])
        self.assertGreater(replay["windows_checked"], 5)

    def test_tail_safe_parse_skips_compaction_wrappers_and_preserves_head_intent(self):
        """Large JSONL starting with compaction wrapper recovers real opening intent, not placeholder."""
        trace = self.root / "long_compacted.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"content": "This session is being continued from a previous conversation that ran out of context."}}),
            json.dumps({"type": "user", "message": {"content": "<command-name>/compact</command-name>"}}),
            json.dumps({"type": "user", "message": {"content": "INTENT: Add hybrid lexical+dense symbol grounding to 01_see_the_math_m2.py"}}),
        ]
        filler = "x" * 600
        for i in range(180):
            lines.append(json.dumps({"type": "assistant", "message": {"content": f"Checked module_{i}.py {filler}"}}))
        trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

        parsed = watcher.parse_jsonl_transcript(trace, max_lines=40, max_tail_bytes=32 * 1024)
        self.assertIn("Add hybrid lexical+dense symbol grounding", parsed["intent"])
        self.assertFalse(parsed["intent_is_placeholder"])
        self.assertLessEqual(len(parsed["steps"]), 40)

    def test_hermes_null_and_iso_timestamps_sort_cleanly(self):
        """NULL and ISO-8601 timestamps in state.db must not crash cross-source sorting."""
        hermes = self.root / "hermes_ts"
        (hermes / "sessions").mkdir(parents=True)
        (hermes / "sessions" / "file_sess.jsonl").write_text(
            json.dumps({"role": "assistant", "content": "Edited worker.py"}) + "\n", encoding="utf-8"
        )
        db_path = hermes / "state.db"
        with closing(sqlite3.connect(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT);
                CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                    content TEXT, tool_calls TEXT, tool_name TEXT, active INTEGER DEFAULT 1);
                INSERT INTO sessions VALUES ('null-ts', NULL, NULL);
                INSERT INTO sessions VALUES ('iso-ts', '2026-11-23T12:00:00Z', NULL);
                INSERT INTO messages(session_id, role, content) VALUES ('null-ts', 'assistant', 'Patched a.py');
                INSERT INTO messages(session_id, role, content) VALUES ('iso-ts', 'assistant', 'Patched b.py');
                """
            )
            conn.commit()
        with patch.object(Path, "home", return_value=self.root):
            targets = watcher.discover_recent_transcripts(10, hermes_home=hermes)
        self.assertEqual(len(targets), 3)

    def test_hybrid_claim_grounding_retains_exact_symbol_rank1(self):
        """Hybrid RRF+lexical retrieval keeps exact symbol hits at Rank #1 even if dense scores are noisy."""
        repo = self.root / "mini_repo"
        repo.mkdir()
        (repo / "alpha.py").write_text(
            "def unrelated_helper_one(x):\n    return x + 1\n\n"
            "def validate_signature_window(payload, secret):\n    return bool(payload and secret)\n",
            encoding="utf-8",
        )
        (repo / "beta.py").write_text(
            "def unrelated_helper_two(y):\n    return y * 2\n",
            encoding="utf-8",
        )
        # Force dense scores to be flat zeros so we prove exact-symbol lexical fusion recovers Rank #1.
        with patch.object(self.embedder, "encode", side_effect=lambda texts: np.zeros((len(texts), 384), dtype=np.float32)):
            res = math_bench.run_repo_claim_grounding(self.embedder, repo)
        self.assertEqual(res["hybrid_rank"], 1)
        self.assertEqual(res["lexical_rank"], 1)
        self.assertEqual(res["caught_rank"], 1)
        self.assertEqual(res["grounding_scope"], "candidate_evidence_for_inspection_not_entailment")

    def test_tiny_jepa_prefers_human_labels_and_chronological_split(self):
        """Tiny JEPA must prefer human_label over heuristic proxy targets and support chronological holdout."""
        episodes = []
        for i in range(10):
            episodes.append(
                {
                    "timestamp_utc": f"2026-11-{10 + i:02d}T10:00:00Z",
                    "current_state": {"summary": f"Inspecting mod_{i}.py"},
                    "intended_outcome": {"summary": f"Fix mod_{i}.py"},
                    "end_state": {"summary": f"Edited mod_{i}.py and passed tests", "thrash_score": 0.95},
                    "assessment_vs_expectations": {
                        "actual_risk_target": 0.95,
                        "human_label": "fp" if i % 2 == 0 else None,
                    },
                }
            )
        targets, human_count, proxy_count = jepa_mod.resolve_risk_targets(episodes)
        self.assertEqual(human_count, 5)
        self.assertEqual(proxy_count, 5)
        self.assertAlmostEqual(float(targets[0]), 0.0)
        self.assertAlmostEqual(float(targets[1]), 0.95)


if __name__ == "__main__":
    unittest.main()
