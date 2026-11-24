"""Offline fixtures: env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -v"""
import importlib
from contextlib import closing
import json
import os
import site
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
watcher = importlib.import_module("02_watch_frontier_agents")
hermes_adapter = importlib.import_module("hermes_transcripts")
DatabaseSession, open_readonly = hermes_adapter.DatabaseSession, hermes_adapter.open_readonly


class HermesAdapterTest(unittest.TestCase):
    def setUp(self):
        # Keep all fixtures in the checkout, not the host's live agent homes.
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        # '?' is illegal in Windows NTFS paths; still test spaces and '#' on Windows, plus '?' on POSIX.
        self.hermes = self.home / ("custom # hermes" if os.name == "nt" else "custom #? hermes")
        self.records = [
            {"role": "session_meta", "content": "metadata is not a step"},
            {"role": "system", "content": "system is not a step"},
            {"role": "user", "content": "Fix retry behavior in worker.py"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"worker.py"}'}}]},
            {"role": "tool", "content": "Created file."},
            {"role": "assistant", "content": "Patched retry timeout in worker.py"},
            {"role": "assistant", "content": "Tested timeout behavior in test_worker.py"},
        ]
        self.jsonl = self.write_jsonl("profiles/alpha/sessions/shared.jsonl")
        self.write_jsonl("profiles/beta/sessions/beta.jsonl")
        self.write_jsonl("sessions/default.jsonl")
        self.write_jsonl("logs/sessions/legacy.jsonl")
        self.database = self.hermes / "profiles/alpha/state.db"
        with closing(sqlite3.connect(self.database)) as conn:
            conn.executescript("""
                CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, ended_at REAL);
                CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                    content TEXT, tool_calls TEXT, tool_name TEXT, active INTEGER DEFAULT 1);
                INSERT INTO sessions VALUES ('shared', 1, NULL), ('db-only', 2, NULL), ('empty', 3, NULL);
            """)
            for session_id in ("shared", "db-only"):
                for record in self.records:
                    conn.execute("INSERT INTO messages(session_id,role,content,tool_calls) VALUES(?,?,?,?)",
                                 (session_id, record["role"], record.get("content"),
                                  json.dumps(record.get("tool_calls", []))))
            conn.execute("INSERT INTO messages(session_id,role,content,active) VALUES('db-only','assistant','obsolete step',0)")
            conn.commit()

    def write_jsonl(self, relative):
        path = self.hermes / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in self.records) + '\nnull\n[]\n{partial', encoding="utf-8")
        return path

    def discover(self):
        with patch.object(Path, "home", return_value=self.home):
            return watcher.discover_recent_transcripts(40, self.hermes)

    def test_discovery_profiles_default_legacy_and_database_dedup(self):
        targets = self.discover()
        self.assertEqual(len(targets), 5)
        self.assertEqual({s for s, _ in targets}, {"hermes", "hermes:alpha", "hermes:beta"})
        databases = [p for _, p in targets if isinstance(p, DatabaseSession)]
        self.assertEqual([p.session_id for p in databases], ["db-only"])
        self.assertEqual(len(watcher.discover_recent_transcripts(1, self.hermes)), 1)
        self.assertEqual(watcher.discover_recent_transcripts(0, self.hermes), [])

    def test_jsonl_tools_and_metadata(self):
        parsed = watcher.parse_jsonl_transcript(self.jsonl)
        self.assertEqual(parsed["intent"], self.records[2]["content"])
        self.assertEqual(len(parsed["steps"]), 3)
        self.assertIn("tool:read_file", parsed["steps"][0])
        self.assertIn("worker.py", parsed["touched_files"])
        self.assertEqual(parsed["filtered_acks"], 1)

    def test_sqlite_and_jsonl_same_steps(self):
        parsed = watcher.parse_jsonl_transcript(DatabaseSession(self.database, "db-only"))
        expected = watcher.parse_jsonl_transcript(self.jsonl)
        for key in ("intent", "steps", "touched_files", "filtered_acks"):
            self.assertEqual(parsed[key], expected[key])
        self.assertEqual(parsed["line_count"], len(self.records))
        self.assertEqual(len(watcher.parse_jsonl_transcript(DatabaseSession(self.database, "db-only"), 2)["steps"]), 2)

    def test_readonly_no_source_mutation_or_database_creation(self):
        before = {p: p.read_bytes() for p in self.hermes.rglob("*") if p.is_file()}
        for _, target in self.discover():
            watcher.parse_jsonl_transcript(target)
        with open_readonly(self.database) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM messages")
        with self.assertRaises((sqlite3.OperationalError, FileNotFoundError)):
            with open_readonly(self.hermes / "missing.db"):
                self.fail("must not create a database")
        after = {p: p.read_bytes() for p in self.hermes.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_wal_commits_are_visible_readonly(self):
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("INSERT INTO messages(session_id,role,content) VALUES('db-only','assistant','Fresh WAL update in worker.py')")
            writer.commit()
            self.assertTrue(Path(str(self.database) + "-wal").exists())
            before = {p: p.read_bytes() for p in self.hermes.rglob("*") if p.is_file()}
            parsed = watcher.parse_jsonl_transcript(DatabaseSession(self.database, "db-only"))
            self.assertEqual(parsed["steps"][-1], "Fresh WAL update in worker.py")
            after = {p: p.read_bytes() for p in self.hermes.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_concurrent_source_changes_reject_capture_and_cleanup(self):
        original = hermes_adapter._copy_database_file
        copies = []

        def changing_copy(source, destination):
            original(source, destination)
            copies.append(source)
            # Test-only concurrent writer; the reader itself never writes sources.
            with closing(sqlite3.connect(self.database)) as writer:
                writer.execute("UPDATE sessions SET started_at=started_at+1")
                writer.commit()

        before_dirs = set(ROOT.glob(".hermes-read-*"))
        with patch.object(hermes_adapter, "_copy_database_file", side_effect=changing_copy):
            with self.assertRaisesRegex(sqlite3.OperationalError, "three snapshot"):
                with open_readonly(self.database):
                    self.fail("unstable capture must not be evaluated")
        self.assertEqual(len(copies), 3)
        self.assertEqual(before_dirs, set(ROOT.glob(".hermes-read-*")))

    def test_closed_wal_store_cannot_create_sidecars(self):
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
        self.assertFalse(Path(str(self.database) + "-wal").exists())
        self.assertFalse(Path(str(self.database) + "-shm").exists())
        before = self.database.read_bytes()
        with open_readonly(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 3)
        self.assertEqual(before, self.database.read_bytes())
        self.assertFalse(Path(str(self.database) + "-wal").exists())
        self.assertFalse(Path(str(self.database) + "-shm").exists())

    def test_missing_home_and_unreadable_store(self):
        with patch.object(Path, "home", return_value=self.home):
            self.assertEqual(watcher.discover_recent_transcripts(10, self.home / "missing"), [])
        self.database.write_bytes(b"not sqlite")
        with self.assertWarns(UserWarning):
            self.assertEqual(len(self.discover()), 4)

    def test_existing_claude_codex_paths_and_explicit_parser(self):
        for path in (self.home / ".claude/projects/p/main.jsonl",
                     self.home / ".claude/projects/p/main/subagents/agent-1.jsonl",
                     self.home / ".codex/sessions/2026/11/21/rollout-1.jsonl"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"type": "assistant", "message": {"content": "Patched worker.py"}}))
        self.assertTrue({"claude_main", "claude_subagent", "codex"}.issubset({s for s, _ in self.discover()}))
        self.assertEqual(watcher.parse_jsonl_transcript(self.home / ".codex/sessions/2026/11/21/rollout-1.jsonl")["steps"], ["Patched worker.py"])

    def test_all_cli_entrypoints_custom_home(self):
        env = dict(os.environ, HOME=str(self.home), USERPROFILE=str(self.home))
        env.setdefault("PYTHONUSERBASE", site.getuserbase())
        env.pop("PYTHONPATH", None)
        for script, flags in [
            ("04_xray_my_sessions.py", ["--max-sessions", "40", "--json"]),
            ("02_watch_frontier_agents.py", ["--once", "--json", "--max-sessions", "40", "--episodes", str(self.home / "episodes.jsonl")]),
            ("self_tap.py", []),
        ]:
            with self.subTest(script=script):
                run = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--force-hash",
                                      "--hermes-home", str(self.hermes), *flags],
                                     env=env, cwd=ROOT, capture_output=True, text=True, check=True)
                data = json.loads(run.stdout)
                rows = data.get("results", data.get("top_sessions", data.get("discovered_transcripts")))
                self.assertEqual(len(rows), 5)
                self.assertEqual({r["source"] for r in rows}, {"hermes", "hermes:alpha", "hermes:beta"})


if __name__ == "__main__":
    unittest.main()
