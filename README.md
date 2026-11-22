# Steppewatch

**Zero-token client-accelerator supervision and prediction-residual JEPA data collection for frontier and local AI agent fleets.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](requirements-m2.txt)
[![Client Silicon: NPU · Apple M-Series](https://img.shields.io/badge/Client%20Silicon-NPU%20%7C%20Apple%20M--Series-111827)](PAPER.md)
[![Prior-Art Release](https://img.shields.io/badge/Release-v1.0--prior--art--20261121-16a34a)](PAPER.md)

**Author:** Derek Ciula · Steppe Integrations · **Date:** November 21, 2026

> **Start here:**
> - **Human builder on Apple Silicon (M2/M3/M4)?** Jump to [`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md) and see real numbers on your own hardware in 5 minutes (`$0` API cost).
> - **AI Agent (Claude Code, Codex CLI, Hermes, or subagent) self-tapping this repo?** Read [`AGENTS.md`](AGENTS.md) and run `python3 scripts/self_tap.py --repo .` for machine-readable JSON onboarding.
> - **Looking for the formal architecture & empirical benchmarks?** Read [`PAPER.md`](PAPER.md) (`v1.1`, live hardware receipts verified `2026-11-21`).

---

## The Problem (and the Idle-Silicon Fix)

When you run remote frontier coding agents and parallel subagents (Claude Code, Codex CLI, Hermes, etc.), two failure modes quietly burn your tokens and your time:

1. **Silent / Polite Stall Loops:** Agents look busy while paraphrasing *"let me double-check the integration plan before editing"* five turns in a row with zero new files or code targets touched.
2. **Unverified Claims Wearing Verification Formatting:** Agents assert a test is missing when it already ships in the file they read, cite deprecated local ports, or drift line citations by 8–25 lines — dumping manual verification onto you.

Hiring *another* frontier LLM to poll and babysit your subagents doubles token burn and latency. Meanwhile, your laptop's own matrix silicon — an **Apple Silicon Neural Engine / Unified Memory Metal GPU** or an **Intel/AMD laptop NPU** — sits at **0% utilization**.

**Steppewatch** turns that idle local silicon into a dual-loop lookout at **`$0.00` extra API cost**:

![Steppewatch Dual-Loop Client-Silicon Supervision and JEPA Residual Architecture](docs/architecture/steppewatch-dual-loop.svg)

---

## Verified Empirical Highlights (Live Hardware — November 21, 2026)

Full methodology, honest misses, and cryptographic receipt digests live in [`PAPER.md`](PAPER.md):

| Capability Measured (Intel Arrow Lake NPU, OpenVINO INT8/FP16) | Measured Result | Latency | Extra LLM Tokens |
|---|---|---:|---:|
| **Discrete GPU VRAM contention** | **`0.0 / 0.0 MiB`** (`verified_zero_gpu_fd_binding = true`) | — | `0` |
| **Pinned `262,144`-col INT8 slab capacity** (`96.5 MiB`, `0` tensor rebinds) | `89` files (`848` code + `.md` syms) = **`0.92%` occupied** | **`1.57 s`** warm (`6.50 s` cold) | `0` |
| **Post-flight substance error catch** (vs. independent frontier audit) | **`3 / 3` real substance errors caught** (Rank #1–#2) | **`34–75 ms`** | `0` |
| **Displaced line-anchor detection** (`23/24` drifted by `-8..-25` lines) | **`21 / 26` flagged outside cited span** | **`54–77 ms`** | `0` |
| **Silent polite-stall separation** (`/detect_thrash`) | Productive **`0.0000`** vs. Polite stall **`0.9278`** | **`26.6–27.7 ms`** | `0` |
| **Parallel tool-ack false-positive calibration** (Proof Step B) | Filtering bare `"Created file."` stubs: `0.72` -> **`0.0000`** | `< 3 ms` | `0` |

> **Explicit Verification Boundary:** Zero-VRAM NPU slab indexing, post-flight claim grounding, silent-loop thrash separation, parallel tool-ack calibration, and `NS_EPISODES`/`NS_TRAJECTORY` partitioning are live-verified. Multi-week longitudinal JEPA predictor training curves across hundreds of real production trajectories are ongoing evaluation and explicitly not claimed yet.

---

## 5-Minute Quick Start (Humans)

Zero API keys required. Runs immediately with `numpy` alone (`--force-hash`), or with real `all-MiniLM-L6-v2` embeddings + Apple Silicon Metal (`MLX` / `MPS`) via `requirements-m2.txt`:

```bash
git clone https://github.com/djcdevelopment/steppewatch.git
cd steppewatch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-m2.txt

# 1. See the math on your own machine (CPU vs. MLX/MPS MatMul, thrash separation, repo claim check)
python3 scripts/01_see_the_math_m2.py --repo /path/to/any/local/repo

# 2. Watch active Claude Code (+ subagents), Codex CLI, or Hermes JSONL transcripts read-only
python3 scripts/02_watch_frontier_agents.py --once

# 3. Verify the tiny local JEPA 5-field residual training loop on synthetic Day-1 episodes
python3 scripts/03_train_tiny_jepa.py --demo-synthetic
```

---

## Agent Self-Tap (Let Your Frontier Agents Wire Themselves In)

Point Claude Code, Codex CLI, Hermes, or any coding agent directly at this repo with:

> *"Read `AGENTS.md` in `steppewatch` and self-tap this workspace so you can supervise subagents and log local JEPA residual episodes."*

An agent can self-tap safely in **3 non-destructive steps** (full protocol in [`AGENTS.md`](AGENTS.md)):
1. **Probe & verify local silicon (JSON output):** `python3 scripts/self_tap.py --repo .`
2. **Adopt the checkpoint ruleset:** Copy [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) into `CLAUDE.md`, `AGENTS.md`, `.claude/skills/steppewatch/SKILL.md`, or Hermes skills.
3. **Supervise subagents & log 5-field episodes (JSON output):** `python3 scripts/02_watch_frontier_agents.py --once --json`

Everything runs **read-only against local transcripts**, touches **zero API keys**, and never intercepts network traffic.

---

## Pairs Well With NVIDIA `auto-ontology`

If you use NVIDIA's Apache-2.0 open-source [**`NVIDIA/auto-ontology`**](https://github.com/NVIDIA/auto-ontology) for discovering and human-certifying domain definitions across schemas and docs, Steppewatch plugs cleanly into both ends:

- **Certified Catalog -> Zero-Token Guardrails:** Load human-certified definitions from `NVIDIA/auto-ontology` into your local `NS_ONTOLOGY` slab so your Mac or NPU checks frontier agent claims against them in milliseconds at `$0.00` token cost — even when an agent skips an MCP query.
- **Residual Surprises -> Human Certification Queue:** Feed recurring high-surprise failures from Steppewatch's `episodes/transitions.jsonl` back into `auto-ontology`'s human-in-the-loop review workflow so real agent mistakes continuously tighten your certified catalog.

---

## Repository Map

| File | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Agent Self-Tap Protocol** — Machine-first guide for Claude Code, Codex CLI, Hermes, and subagents to bootstrap, self-check, and supervise without burning extra frontier tokens. |
| [`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md) | **Apple M2 Builder's Cookbook** — Tailored to 24 GB unified-memory Macs (`Mac14,7`), explaining ANE vs. `MLX` Metal GPU physics, transcript paths, and local JEPA data collection. |
| [`PAPER.md`](PAPER.md) | **Technical Preprint (`v1.1`)** — Formal architecture, math, live Arrow Lake NPU benchmarks (Proof A & B), Apple Silicon mapping, and `NVIDIA/auto-ontology` interop. |
| [`docs/EXPLAINER-BRIEF-NOTEBOOKLM.md`](docs/EXPLAINER-BRIEF-NOTEBOOKLM.md) | **Explainer Video / Audio Source Brief** — Clean narrative brief ready to drop straight into NotebookLM for generating a short explainer video or podcast overview. |
| [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) | **Drop-In Starter Ruleset / Skill** — `INTENT:`, `STUCK:`, grounded-claims-only, compact `END_STATE:` receipt, and disjoint subagent targets. |
| [`scripts/self_tap.py`](scripts/self_tap.py) | **One-Command Agent Self-Tap Probe** — Detects hardware/backends, finds active Claude/Codex/Hermes logs, runs a fast self-check, and emits structured JSON onboarding instructions. |
| [`scripts/01_see_the_math_m2.py`](scripts/01_see_the_math_m2.py) | **Hardware Benchmark & Proof** — Batched `[Q, 384] @ [384, 262144]` slab timing, thrash separation, and local repo claim grounding. |
| [`scripts/02_watch_frontier_agents.py`](scripts/02_watch_frontier_agents.py) | **Live Read-Only Watcher (`--json` supported)** — Tails Claude Code (+ subagents), Codex, and Hermes JSONL logs, filters bare tool-ack stubs, flags loops/stalls, and logs 5-field episodes. |
| [`scripts/03_train_tiny_jepa.py`](scripts/03_train_tiny_jepa.py) | **Tiny Local JEPA Trainer** — Trains a compact `384-D` latent state + stall-risk MLP predictor on collected episodes (`--demo-synthetic` supported) and saves a `.npz` checkpoint. |
| [`MANIFEST.sha256`](MANIFEST.sha256) | Cryptographic SHA-256 provenance manifest for all repository files. |
