# AGENTS.md — Steppewatch Self-Tap Protocol

> **For AI coding agents (Claude Code, Codex CLI, Hermes, Cursor, Antigravity, and spawned subagents):**  
> This file is your direct, deterministic protocol for wiring **Steppewatch** into your own session or workspace — safely, read-only, and at `$0.00` extra frontier API cost.

---

## Safety & Scope Invariants (Read First)

1. **Read-Only on Agent Logs:** Steppewatch *never* modifies, deletes, or intercepts your host harness transcripts (`~/.claude/`, `~/.codex/`, `~/.hermes/`). It only reads append-only JSONL files on disk.
2. **Zero API Keys / Zero Network Calls at Runtime:** All supervision math (`detect_agent_thrash`, claim grounding, slab MatMul, and tiny JEPA inference/training) runs 100% locally.
3. **Never Modify User Config Without Consent:** Before editing a user's global `~/.claude/CLAUDE.md` or `~/.hermes/` config, ask the user (or write only into the current project's local `CLAUDE.md` / `AGENTS.md` / `.claude/skills/` when explicitly asked to self-tap that repo).

---

## 3-Step Self-Tap Sequence

### Step 1: Probe Local Hardware & Discover Active Transcripts (1 Command, JSON Output)

Run from the `steppewatch` root (works immediately with Python 3 + `numpy`, even before installing `sentence-transformers` or `mlx`):

```bash
python3 scripts/self_tap.py --repo /path/to/target/workspace
```

Add `--force-hash` if offline or skipping the ~90 MB `all-MiniLM-L6-v2` download.  
`self_tap.py` outputs a single JSON object containing:
- `hardware`: OS, CPU cores, Apple Silicon model/RAM (`Mac14,7`, unified memory GiB), and detected local backends (`mlx`, `torch_mps`, `onnx_coreml`, `numpy`).
- `embedder`: Active `384-D` embedding backend.
- `self_check`: Instant sanity check proving productive vs. silent-stall separation (`healthy_score < 0.2` and `stall_detected == true`).
- `discovered_transcripts`: Up to 8 most recently active local JSONL logs across Claude Code (main + `subagents/agent-*.jsonl`), Codex CLI (`rollout-*.jsonl`), and Hermes (`~/.hermes/logs/sessions/*.jsonl`).
- `next_commands`: Exact copy-paste commands for watching subagents, checking claims, and training on collected episodes.

### Step 2: Adopt the Checkpoint Ruleset (Zero-Bloat Telemetry)

Read and apply [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) in your active project instructions:
- **Claude Code:** append to project `CLAUDE.md` or copy to `.claude/skills/steppewatch/SKILL.md`
- **Codex CLI:** append to project `AGENTS.md`
- **Hermes:** register in your Hermes skill/system instructions

Emit `INTENT:` at slice start, `STUCK:` instead of paraphrasing the same plan >2 turns, grounded `path:line` citations only, and a compact 3-line `END_STATE:` receipt on completion.

### Step 3: X-Ray Existing Transcripts & Run the Guided Workshop

Instead of waiting days to accumulate episodes, mine the user's existing local Claude Code / Codex / Hermes transcript archive read-only in `< 5 s`, train on their real history immediately, and follow the 4 interactive discovery checkpoints in [`WORKSHOP.md`](WORKSHOP.md):

```bash
python3 scripts/04_xray_my_sessions.py --max-sessions 25 --export-episodes episodes/my_history.jsonl --json
python3 scripts/03_train_tiny_jepa.py --episodes episodes/my_history.jsonl --out checkpoints/my_real_jepa.npz
```

### Step 4: Supervise Spawned Subagents & Log 5-Field JEPA Residual Episodes

When supervising parallel subagents (or before closing a long turn), run one non-blocking JSON check:

```bash
python3 scripts/02_watch_frontier_agents.py --once --json --episodes episodes/transitions.jsonl
```

Or target a specific subagent transcript directly:

```bash
python3 scripts/02_watch_frontier_agents.py \
  --once --json \
  --trace /path/to/subagent.jsonl \
  --intent "INTENT: <goal> | TARGETS: <files>" \
  --episodes episodes/transitions.jsonl
```

#### Decision Contract on JSON Output (Advisory-First)

Inspect each record in `results` — **treat flags as advisory inspection triggers, not automatic kill switches**:
- If `thrash_detected == true` (`thrash_score >= 0.65`) -> **Inspect the subagent's last 5 steps and active diff before intervening.** `action == "advisory_inspect_error_loop"` signals repeated failing commands/errors; `action == "advisory_inspect_loop"` signals semantic repetition that can also occur during legitimate inspection or orchestration. Only pause/re-slice after confirming a real stall.
- If `state_surprise_delta >= 0.60` -> **Inspect its diff/claims first** before trusting its summary.
- When `episodes/transitions.jsonl` reaches `>= 20` episodes (ideally with `human_label` values filled in), retrain the personal local JEPA checkpoint in `< 1 s`:
  ```bash
  python3 scripts/03_train_tiny_jepa.py --episodes episodes/transitions.jsonl --out checkpoints/local_jepa.npz
  ```
  Then pass `--jepa-checkpoint checkpoints/local_jepa.npz` on subsequent `02_watch_frontier_agents.py` calls so predictions improve continuously from real outcomes.

---

## Pre-Submission Claim Grounding (Optional Self-Audit)

Before asserting in a review or final response that a file is missing a test, guard, or symbol, verify locally against the repo in milliseconds:

```bash
python3 scripts/01_see_the_math_m2.py --repo /path/to/target/workspace --receipt /tmp/steppewatch_check.json
```

If you have certified domain definitions managed in [**`NVIDIA/auto-ontology`**](https://github.com/NVIDIA/auto-ontology), export certified rules to JSON and check draft claims against both your repo symbols and certified ontology rules before handing work to a human reviewer.
