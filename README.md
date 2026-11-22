# Steppewatch

**Zero-token client-accelerator supervision and prediction-residual JEPA data collection for frontier and local AI agent fleets.**

**Author:** Derek Ciula · Steppe Integrations  
**Date:** November 21, 2026 · **License:** MIT

---

## Why Steppewatch

While remote frontier agents (Claude Code + subagents, Codex CLI, Hermes, etc.) think, loop, drift out of scope, or mis-cite files and lines, your laptop's local matrix silicon — an Apple Silicon Neural Engine / Unified Memory Metal GPU or an Intel/AMD laptop NPU — sits idle.

**Steppewatch** turns that idle local silicon into a zero-API-cost lookout that does two connected jobs:
1. **Supervises frontier agents & subagents in real time** (catching both loud error loops and silent paraphrased stall loops, and grounding claims against your real codebase in milliseconds at `$0.00` token cost).
2. **Collects 5-field Prediction-Residual Episodes (`Δ = y − ŷ`)** from every watched run so you can train a tiny local **JEPA (Joint Embedding Predictive Architecture)** watchdog tuned to your own repositories and agents.

---

## Repository Contents

| File | What It Does |
|---|---|
| [`PAPER.md`](PAPER.md) | **Technical Preprint & Prior-Art Record (`v1.1`, 2026-11-21)** — Architecture, mathematical formulation, live Intel Arrow Lake NPU measurements (`0.0 MiB` GPU VRAM, `3/3` substance errors caught in `34–75 ms`, `21/26` displaced line spans flagged, Proof Step B warm-indexing & live-watchdog calibration), honest verification boundaries, and Apple Silicon portability. |
| [`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md) | **Apple M2 Builder's Cookbook (`Mac14,7`, 24 GB Unified Memory, macOS 26.5.2)** — Hands-on guide for builders running Claude Code (+ subagents), Codex CLI, and Hermes: see the math on your own Mac in 5 minutes, watch live local transcripts, and collect + train on your own JEPA residual episodes. |
| [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) | **Drop-In Starter Ruleset / Skill** — Ready for `CLAUDE.md`, `AGENTS.md`, or Hermes skills (`INTENT:`, `STUCK:`, grounded-claims-only, compact `END_STATE:` receipt, disjoint subagent targets). |
| [`scripts/01_see_the_math_m2.py`](scripts/01_see_the_math_m2.py) | **Step 1: See the Math on Your Hardware** — Benchmarks local CPU vs. Apple `MLX` Metal GPU / `PyTorch MPS` `[Q, 384] @ [384, N]` slabs up to `262,144` columns, demonstrates productive vs. polite-stall thrash detection, and runs zero-token claim grounding against any local repo. |
| [`scripts/02_watch_frontier_agents.py`](scripts/02_watch_frontier_agents.py) | **Step 2: Live Read-Only Watcher & Episode Collector** — Tails local Claude Code (main + subagents), Codex CLI, and Hermes JSONL transcripts (filtering bare parallel tool-ack stubs), alerts on stalls/loops, and appends 5-field residual episodes to `episodes/transitions.jsonl`. |
| [`scripts/03_train_tiny_jepa.py`](scripts/03_train_tiny_jepa.py) | **Step 3: Tiny Local JEPA Trainer** — Trains a compact `384-D` latent state + stall-risk predictor on your collected episodes (includes `--demo-synthetic` for Day-1 testing) and exports a `.npz` checkpoint loaded live by `02_watch_frontier_agents.py`. |
| [`MANIFEST.sha256`](MANIFEST.sha256) | Cryptographic SHA-256 provenance manifest for every file in this release. |

---

## 5-Minute Quick Start (Zero API Keys Required)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-m2.txt

# 1. See the math on your own machine (benchmarks + thrash detection + local repo claim grounding)
python3 scripts/01_see_the_math_m2.py --repo /path/to/any/local/repo

# 2. Watch your active Claude Code / Codex / Hermes transcripts read-only (0 frontier tokens)
python3 scripts/02_watch_frontier_agents.py --once

# 3. Verify the local JEPA residual-episode training loop on synthetic Day-1 episodes
python3 scripts/03_train_tiny_jepa.py --demo-synthetic
```

Start with [`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md) to build immediately, or [`PAPER.md`](PAPER.md) for the formal technical paper and measured benchmarks.

---

## Pairs Well With NVIDIA `auto-ontology`

If you already use (or want to try) NVIDIA's Apache-2.0 open-source [**`NVIDIA/auto-ontology`**](https://github.com/NVIDIA/auto-ontology) for discovering and human-certifying domain definitions across schemas and docs, Steppewatch plugs directly into both ends of that workflow:

- **Certified Rules -> Local Guardrails:** Load certified definitions exported from `NVIDIA/auto-ontology` into your local `NS_ONTOLOGY` vector slab so your Mac or NPU checks frontier agent claims against them in milliseconds at `$0.00` token cost — even when an agent skips an MCP query.
- **Residual Surprises -> Human Review:** Feed recurring high-surprise failures from Steppewatch's 5-field `episodes/transitions.jsonl` back into `auto-ontology`'s human-in-the-loop certification workflow so every real agent mistake tightens your certified catalog.

