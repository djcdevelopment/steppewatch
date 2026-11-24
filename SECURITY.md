# Security & Privacy Posture

Steppewatch is designed as an **offline, local-first, read-only** supervision toolkit for your own machine.

## Core Guarantees

1. **Zero Network Calls at Runtime:**
   - Core scripts (`--force-hash` mode and the `.NET` core CLI) perform **zero** network requests and require **zero** API keys.
   - Optional local embedding/reranking models (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) are downloaded once from Hugging Face only when you explicitly run without `--force-hash` (or pass `--download-model` in `.NET` ONNX CLI). Pass `--offline` or set `HF_HUB_OFFLINE=1` to strictly prohibit network fetches and load from cache only.
2. **Read-Only on Host Agent Logs & Configs:**
   - Steppewatch never modifies, truncates, or deletes Claude Code (`~/.claude/`), Codex CLI (`~/.codex/`), or Hermes (`~/.hermes/`) transcripts or global configuration files.
   - For Hermes `state.db` SQLite fallback, Steppewatch never opens live source databases directly through SQLite (preventing `-wal`/`-shm` sidecar creation or mtime changes). Instead, it copies/APFS-clones `state.db` (+ `-wal` if present) into a private `0700` temporary capture, verifies file identity/size/timestamps before and after copying, opens the private capture with `mode=ro` + `PRAGMA query_only=ON`, and deletes the capture on exit.
3. **Advisory-Only Supervision:**
   - Heuristic stall/loop flags (`advisory_inspect_error_loop`, `advisory_inspect_loop`) are advisory inspection triggers only. Steppewatch never kills processes or terminates agents automatically.

## Handling Your Local Transcripts & Episodes

- `episodes/`, `checkpoints/`, and `receipts/` are listed in `.gitignore` so exported residual episodes and local session summaries are never accidentally committed.
- **Never include real private transcripts, prompts, API keys, or personal paths** in GitHub issues, pull requests, or test fixtures. All repository tests use synthetic offline fixtures only.

## Reporting a Vulnerability

If you discover a safety bug (such as unexpected source-file mutation, unsafe path handling, or accidental data exposure), please open a minimal reproduction issue or contact `derek@steppeintegrations.com` directly before disclosing sensitive exploit details publicly.
