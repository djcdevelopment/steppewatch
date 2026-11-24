"""Read-only discovery of Hermes default and named-profile session stores."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import sqlite3
import warnings
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class DatabaseSession:
    database: Path
    session_id: str

    @property
    def name(self) -> str:
        return self.session_id

    @property
    def stem(self) -> str:
        return self.session_id

    def __str__(self) -> str:
        return f"{self.database}#session={self.session_id}"


Transcript = Union[Path, DatabaseSession]


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    except FileNotFoundError:
        return ()


def _coerce_timestamp(value: Any, fallback: float) -> float:
    """Coerce SQLite numeric, ISO-8601 string, or NULL timestamps into sortable epoch floats."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return fallback
        try:
            return float(raw)
        except ValueError:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return fallback
    return fallback


def _copy_database_file(source: Path, destination: Path) -> None:
    # APFS clones avoid copying multi-GB profile stores on the target Mac.
    if sys.platform == "darwin":
        result = subprocess.run(["/bin/cp", "-c", str(source), str(destination)],
                                capture_output=True, check=False)
        if result.returncode == 0:
            return
    if source.stat().st_size > 256 * 1024 * 1024:
        raise sqlite3.OperationalError("snapshot exceeds 256 MiB without filesystem cloning")
    shutil.copyfile(source, destination)


@contextmanager
def open_readonly(database: Path):
    """Read a stable private DB/WAL capture; never open live SQLite sidecars."""
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    # TemporaryDirectory is private (0700), workspace-local, and removed on exit.
    with tempfile.TemporaryDirectory(prefix=".hermes-read-", dir=Path(__file__).resolve().parents[1]) as tmp:
        target = Path(tmp) / "state.db"
        sources = (database, Path(str(database) + "-wal"))
        for _ in range(3):
            before = [_stamp(path) for path in sources]
            for path, stamp in zip(sources, before):
                destination = target if path == database else Path(str(target) + "-wal")
                destination.unlink(missing_ok=True)
                if stamp:
                    _copy_database_file(path, destination)
            if before == [_stamp(path) for path in sources]:
                break
        else:
            raise sqlite3.OperationalError("database changed during all three snapshot attempts")
        # URI escaping handles spaces, '?' and '#' in custom homes.
        with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
            conn.execute("PRAGMA query_only=ON")
            yield conn


def discover_hermes(home: Path, max_files: int) -> List[tuple]:
    """Prefer JSONL for a session, falling back to state.db for missing transcripts."""
    found = []
    roots = [("hermes", home)]
    roots.extend((f"hermes:{p.name}", p) for p in sorted((home / "profiles").glob("*")) if p.is_dir())
    for source, root in roots:
        seen = set()
        for directory in (root / "sessions", root / "logs" / "sessions"):
            for path in directory.glob("*.jsonl"):
                try:
                    if path.stem not in seen:
                        found.append((path.stat().st_mtime, source, path))
                        seen.add(path.stem)
                except OSError:
                    continue
        database = root / "state.db"
        if not database.is_file():
            continue
        try:
            db_mtime = database.stat().st_mtime
            with open_readonly(database) as conn:
                # Limit per profile before parsing message bodies. EXISTS avoids empty sessions.
                columns = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
                recent = "COALESCE(last_activity_at, ended_at, started_at)" if "last_activity_at" in columns else "COALESCE(ended_at, started_at)"
                rows = conn.execute(
                    f"SELECT id, {recent} FROM sessions "
                    "WHERE EXISTS (SELECT 1 FROM messages WHERE session_id = sessions.id) "
                    f"ORDER BY {recent} DESC LIMIT ?",
                    (max_files + len(seen),),
                )
                count = 0
                for session_id, timestamp in rows:
                    if session_id not in seen:
                        found.append((_coerce_timestamp(timestamp, db_mtime), source, DatabaseSession(database, session_id)))
                        count += 1
                        if count >= max_files:
                            break
        except (sqlite3.Error, OSError) as exc:
            warnings.warn(f"Cannot read Hermes store {database}: {exc}")
    return found


def read_database_session(target: DatabaseSession, max_lines: int) -> tuple:
    """Return the bounded active message tail in transcript order, plus total rows."""
    try:
        with open_readonly(target.database) as conn:
            conn.row_factory = sqlite3.Row
            columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
            active = " AND active = 1" if "active" in columns else ""
            where = " WHERE session_id = ?" + active
            count = conn.execute("SELECT COUNT(*) FROM messages" + where, (target.session_id,)).fetchone()[0]
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_name FROM messages" + where + " ORDER BY id DESC LIMIT ?",
                (target.session_id, max_lines),
            ).fetchall()
            records: List[Dict[str, Any]] = []
            for row in reversed(rows):
                record = dict(row)
                if isinstance(record["tool_calls"], str):
                    try:
                        record["tool_calls"] = json.loads(record["tool_calls"])
                    except ValueError:
                        record["tool_calls"] = []
                records.append(record)
            return records, count
    except (sqlite3.Error, OSError) as exc:
        warnings.warn(f"Cannot read Hermes session {target}: {exc}")
        return [], 0
