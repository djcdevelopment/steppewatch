# Steppewatch

**Your laptop already has matrix silicon sitting at 0% while your cloud AI agents burn tokens. Put it to work for `$0.00`.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](requirements-m2.txt)
[![Client Silicon: NPU · Apple M-Series](https://img.shields.io/badge/Client%20Silicon-NPU%20%7C%20Apple%20M--Series-111827)](PAPER.md)
[![Prior-Art Release](https://img.shields.io/badge/Release-v1.0--prior--art--20261121-16a34a)](PAPER.md)

**Author:** Derek Ciula · Steppe Integrations · **Date:** November 21, 2026

---

## Why Almost Nobody Tells You This

Cloud AI providers bill by the token — whether your coding agent ships working software, gets stuck paraphrasing the same plan five turns in a row, or hires a *second* cloud model to babysit the first one. Almost no one selling cloud APIs has an incentive to remind you how much unused ML compute you already paid for on your own desk.

Open Activity Monitor or Task Manager while Claude Code, Codex CLI, or Hermes is running. On almost every modern machine — an **Apple Silicon MacBook** (unified memory + Metal GPU + 16-core Neural Engine) or a modern **Intel/AMD NPU laptop** — your dedicated matrix hardware is sitting completely idle.

**Steppewatch** shows you how to reclaim that local hardware in 5 minutes — with **zero API keys, zero network interception, and `$0.00` extra token cost** — for two things:
1. **Watch your remote frontier agents & subagents in real time:** Catch silent polite stall loops in milliseconds and check agent claims against your real codebase before you spend your own review time.
2. **Collect your own local JEPA training data:** Turn every run into a 5-field prediction-residual episode so your machine trains a tiny personal watchdog tuned to *your* repos and *your* agents.

![Steppewatch UML Horizontal Swimlane Architecture](docs/architecture/steppewatch-dual-loop.svg)

---

## Pick Your Own Starting Point (You Own the Hardware — Test the Math Yourself)

Nothing here asks you to trust marketing claims. Run one command on your own machine and read your own millisecond numbers:

### Option A — 60-Second Dry Run (Just Python + `numpy`, Zero Model Downloads)
```bash
git clone https://github.com/djcdevelopment/steppewatch.git
cd steppewatch
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
python3 scripts/01_see_the_math_m2.py --force-hash --repo .
```

### Option B — Full 5-Minute Run (Real `all-MiniLM-L6-v2` Embeddings + Apple Silicon `MLX` Metal GPU)
On an Apple M2/M3/M4 MacBook (or Linux/Windows laptop), install the full local stack and point it at any git repo you actually work in:
```bash
git clone https://github.com/djcdevelopment/steppewatch.git
cd steppewatch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-m2.txt

# 1. Benchmark your own CPU vs. Metal GPU (up to 262,144 vectors) + test thrash & claim grounding
python3 scripts/01_see_the_math_m2.py --repo /path/to/any/local/repo

# 2. Inspect your own active Claude Code (+ subagents), Codex CLI, or Hermes logs read-only
python3 scripts/02_watch_frontier_agents.py --once

# 3. Watch a tiny local JEPA predictor train on 5-field residual episodes in < 1 second
python3 scripts/03_train_tiny_jepa.py --demo-synthetic
```
👉 **Running on an Apple M2/M3/M4 Mac?** Follow the step-by-step **[`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md)** — it explains unified memory vs. Neural Engine vs. `MLX` Metal GPU in plain English and shows exactly where Claude Code, Codex CLI, and Hermes already save their local JSONL logs on macOS.

### Option C — Let Your Own Agent Self-Tap & Walk You Through It
Already live inside Claude Code, Codex CLI, or Hermes? Paste this single prompt:

> *"Clone `https://github.com/djcdevelopment/steppewatch.git`, read `AGENTS.md`, run `python3 scripts/self_tap.py --repo .`, and show me what my local hardware can do (don't change any global configs without asking me)."*

Everything runs **read-only on local transcripts**, touches **zero API keys**, and gives you clean JSON output ([`AGENTS.md`](AGENTS.md)).

---

## How to Read Your Own Numbers (No ML PhD Required)

When `01_see_the_math_m2.py` finishes on your machine, you'll see three concrete proofs:
- **Part A (`262,144` columns = `96.5 MiB`):** A single matrix holding hundreds of repos worth of symbols, rules, and history takes less than `0.4%` of a 24 GB Mac's memory and scores `16` queries at once in milliseconds — faster than a single round-trip to a cloud API.
- **Part B (`thrash_score` `0.0000` vs. `0.85+`):** Even when a healthy agent hits a bug mid-run, it touches new symbols and applies real fixes (`score = 0.0`, `continue`). A polite agent that re-reads the same plan 5 times with zero new code targets lights up immediately (`early_abort_and_reroute`) — with `$0` spent on a second LLM.
- **Part C (Claim Grounding):** The script picks a real function from your own repo, fakes an agent claim saying *"this repo is missing `<symbol>`"*, and shows your local silicon finding the exact `file:line` proving the claim false in milliseconds.

---

## Verified Reference Measurements (Intel Arrow Lake NPU — November 21, 2026)

Full methodology, honest misses, and cryptographic receipt digests live in [`PAPER.md`](PAPER.md):

| Capability Measured (OpenVINO INT8/FP16 NPU) | Measured Result | Latency | Extra LLM Tokens |
|---|---|---:|---:|
| **Discrete GPU VRAM contention** | **`0.0 / 0.0 MiB`** (`verified_zero_gpu_fd_binding = true`) | — | `0` |
| **Pinned `262,144`-col INT8 slab** (`96.5 MiB`, `0` rebinds) | `89` files (`848` code + `.md` syms) = **`0.92%` occupied** | **`1.57 s`** warm (`6.50 s` cold) | `0` |
| **Post-flight substance error catch** (vs. frontier audit) | **`3 / 3` real substance errors caught** (Rank #1–#2) | **`34–75 ms`** | `0` |
| **Displaced line-anchor detection** (`23/24` drifted `-8..-25` lines) | **`21 / 26` flagged outside cited span** | **`54–77 ms`** | `0` |
| **Silent polite-stall separation** (`/detect_thrash`) | Productive **`0.0000`** vs. Polite stall **`0.9278`** | **`26.6–27.7 ms`** | `0` |
| **Parallel tool-ack calibration** (Proof Step B) | Filtering bare `"Created file."` stubs: `0.72` -> **`0.0000`** | `< 3 ms` | `0` |

> **Explicit Verification Boundary:** Zero-VRAM NPU slab indexing, post-flight claim grounding, silent-loop thrash separation, parallel tool-ack calibration, and `NS_EPISODES`/`NS_TRAJECTORY` partitioning are live-verified. Multi-week longitudinal JEPA predictor training curves across hundreds of real production trajectories are ongoing evaluation and explicitly not claimed yet.

### Independent Apple M2 Verification (`Mac14,7`, 24 GB Unified Memory — November 21, 2026)

Verified on an independent 2022 13-inch M2 MacBook Pro (`PyTorch MPS` Metal `FP16` `192 MiB` matrix vs. CPU `FP32` `384 MiB` against `262,144 × 384` vectors, Option A zero-download run):

| Batch (`Q` vs. `262,144` vectors) | M2 CPU (`FP32`) | M2 Metal GPU (`FP16`, `192 MiB`) | Speedup |
|---:|---:|---:|---:|
| `Q = 1` | `8.83 ms` | **`8.43 ms`** | `1.05×` |
| `Q = 8` | `34.75 ms` | **`10.14 ms`** | **`3.43×`** |
| `Q = 16` | `46.06 ms` | **`10.20 ms`** | **`4.52×`** |
| `Q = 64` | `64.34 ms` | **`13.71 ms`** | **`4.69×`** |

Scaling from `1` to `16` concurrent queries added only **`+1.77 ms`** on M2 Metal (`8.43 -> 10.20 ms`), while silent-stall detection flagged `0.8635` vs. `0.0000` in **`2.31 ms`** and repo claim grounding indexed `94` symbols in `25.79 ms` (query **`2.78 ms`**). *(Note: `96.5 MiB` is the quantized `INT8 + FP16 scales` slab footprint; the timed Metal matrix above is `192 MiB` in `FP16`.)*

---

## Pairs Well With NVIDIA `auto-ontology`

If you use NVIDIA's Apache-2.0 open-source [**`NVIDIA/auto-ontology`**](https://github.com/NVIDIA/auto-ontology) for discovering and human-certifying domain definitions across schemas and docs, Steppewatch plugs cleanly into both ends:

- **Certified Catalog -> Zero-Token Guardrails:** Load human-certified definitions from `NVIDIA/auto-ontology` into your local `NS_ONTOLOGY` slab so your Mac or NPU checks frontier agent claims against them in milliseconds at `$0.00` token cost — even when an agent skips an MCP query.
- **Residual Surprises -> Human Certification Queue:** Feed recurring high-surprise failures from Steppewatch's `episodes/transitions.jsonl` back into `auto-ontology`'s human-in-the-loop review workflow so real agent mistakes continuously tighten your certified catalog.

---

## Repository Map

| File | Purpose |
|---|---|
| [`COOKBOOK-MAC-M2.md`](COOKBOOK-MAC-M2.md) | **Apple M2 Builder's Cookbook** — Step-by-step guide tailored to 24 GB unified-memory Macs (`Mac14,7`), ANE vs. `MLX` Metal GPU physics, transcript paths, and local JEPA data collection. |
| [`AGENTS.md`](AGENTS.md) | **Agent Self-Tap Protocol** — Machine-first guide for Claude Code, Codex CLI, Hermes, and subagents to bootstrap, self-check, and supervise safely in JSON. |
| [`PAPER.md`](PAPER.md) | **Technical Preprint (`v1.1`)** — Formal architecture, UML sequence diagram, live Arrow Lake NPU benchmarks (Proof A & B), Apple Silicon mapping, and `NVIDIA/auto-ontology` interop. |
| [`docs/EXPLAINER-BRIEF-NOTEBOOKLM.md`](docs/EXPLAINER-BRIEF-NOTEBOOKLM.md) | **Explainer Video / Audio Source Brief** — Narrative source file ready to drop into NotebookLM to generate a 3–5 minute explainer video or podcast overview. |
| [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) | **Drop-In Starter Ruleset / Skill** — `INTENT:`, `STUCK:`, grounded-claims-only, compact `END_STATE:` receipt, and disjoint subagent targets. |
| [`scripts/self_tap.py`](scripts/self_tap.py) | **One-Command Agent Self-Tap Probe** — Detects hardware/backends, finds active Claude/Codex/Hermes logs, runs a fast self-check, and emits structured JSON onboarding instructions. |
| [`scripts/01_see_the_math_m2.py`](scripts/01_see_the_math_m2.py) | **Hardware Benchmark & Proof** — Batched `[Q, 384] @ [384, 262144]` slab timing, thrash separation, and local repo claim grounding. |
| [`scripts/02_watch_frontier_agents.py`](scripts/02_watch_frontier_agents.py) | **Live Read-Only Watcher (`--json` supported)** — Tails Claude Code (+ subagents), Codex, and Hermes JSONL logs, filters bare tool-ack stubs, flags loops/stalls, and logs 5-field episodes. |
| [`scripts/03_train_tiny_jepa.py`](scripts/03_train_tiny_jepa.py) | **Tiny Local JEPA Trainer** — Trains a compact `384-D` latent state + stall-risk MLP predictor on collected episodes (`--demo-synthetic` supported) and saves a `.npz` checkpoint. |
| [`MANIFEST.sha256`](MANIFEST.sha256) | Cryptographic SHA-256 provenance manifest for all repository files. |
