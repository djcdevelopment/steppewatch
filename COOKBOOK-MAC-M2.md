# Steppewatch: Local Silicon Watchdog & JEPA Data Cookbook (Apple M2 Edition)

**Tailored specifically for:** 13-inch MacBook Pro (2022) — Apple M2 (`Mac14,7`, `MNEW3LL/A`)  
**Specs:** 8-core CPU (4P + 4E) · 10-core Metal GPU · 16-core Apple Neural Engine (`~15.8 TOPS`) · **24 GB Unified Memory** (`100 GB/s`) · ~1 TB SSD · macOS `26.5.2 (25F84)`  
**Built for builders using:** Claude Code (+ subagents), Codex CLI, Hermes, and remote frontier agents

---

## The Core Idea (Reclaiming the Compute You Already Own)

If you mostly run remote frontier models (Claude, Codex, Hermes, etc.), look at Activity Monitor while they are working:

- Your cloud agent is burning metered tokens, spawning subagents, reading files, and sometimes quietly getting stuck in a **polite planning loop** (*"Let me double-check the integration plan before editing..."* × 5), drifting out of scope, or hallucinating that a test/function doesn't exist.
- Meanwhile, your **24 GB Apple M2** — its 16-core Neural Engine, 10-core Metal GPU, and `100 GB/s` unified memory — is sitting basically idle.

Cloud providers bill per token whether your agent is making progress or spinning in circles, so almost nobody is incentivized to tell you how much real ML supervision power is sitting untouched on your desk. Most people default to spawning *another* cloud agent to babysit the first one — paying twice for the same mistake.

Instead, **Steppewatch** hands that control back to you using your Mac's local silicon for **two tightly connected jobs — at `$0.00` API cost**:

1. **Watch your frontier agents & subagents in real time:** Tail the local JSONL transcripts that Claude Code, Codex CLI, and Hermes *already write to your SSD*, embed sliding step windows locally (`384-D` vectors), catch both loud error loops and silent paraphrased stall loops in milliseconds, and cross-check agent claims against your real codebase before you spend your own review minutes.
2. **Collect your own local JEPA training data while they work:** Turn every agent run into a compact **5-field residual episode** (`current_state`, `intended_outcome`, `local_prediction`, `end_state`, `outcome_vs_expectation`) so your M2 can train a tiny local **JEPA (Joint Embedding Predictive Architecture)** watchdog that learns *your* repos and *your* frontier agents' failure habits over time.

This cookbook gives you the pieces and lets you verify the math on your own M2 first. Everything else is yours to build on.

---

## Quick Hardware Truth About Your M2 (`Mac14,7`, 24 GB)

Before writing a line of code, one real Apple Silicon fact saves you hours of guesswork:

- **Your 24 GB Unified Memory is a superpower:** CPU, 10-core GPU, and 16-core Neural Engine all read the same `100 GB/s` memory pool with zero PCIe copy penalty. A `262,144`-column `384-D` INT8 matrix of your entire workspace + rules + history is only **`~96.5 MB`** — less than `0.4%` of your RAM!
- **ANE (CoreML) vs. Metal GPU (`MLX` / `MPS`) — know which to use:**
  - **Apple Neural Engine (CoreML)** is ultra-efficient on **fixed static shapes** (e.g., encoding fixed-length chunks at `S=128` while sipping ~1–3W on battery). If you feed it dynamic shapes, CoreML recompiles or silently falls back to CPU.
  - **Apple `MLX` (or PyTorch `MPS`) on your 10-core Metal GPU** crushes **batched matrix multiplies** (`[Q, 384] @ [384, N]`) and **training tiny local JEPA MLPs** right inside unified memory.
- **Don't take anyone's word for it:** Step 1 benchmarks your exact M2 backends side-by-side so you see the real millisecond numbers yourself.

---

## Step 1: See the Math on Your Own Mac (60 Seconds to 5 Minutes, Zero API Keys)

Clone the repo straight from GitHub and create a clean Python environment:

```bash
git clone https://github.com/djcdevelopment/steppewatch.git
cd steppewatch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-m2.txt
```

> **Want a 60-second zero-download dry run first?** Skip `requirements-m2.txt`, run just `pip install numpy`, and pass `--force-hash` (`python3 scripts/01_see_the_math_m2.py --force-hash --repo .`). Installing `requirements-m2.txt` unlocks real `384-D` `all-MiniLM-L6-v2` semantic embeddings and Apple `MLX` Metal GPU acceleration on your M2.
>
> **Prefer to let Claude Code / Codex / Hermes set it up and explain the output?** Paste this into your agent:  
> *"Clone `https://github.com/djcdevelopment/steppewatch.git`, read `AGENTS.md`, run `python3 scripts/self_tap.py --repo .`, and walk me through what my local hardware can do."*

Run the benchmark against any real git repo on your Mac:

```bash
python3 scripts/01_see_the_math_m2.py --repo /path/to/one/of/your/repos
```

### What You Will See (and Why Each Number Matters)

1. **Part A — Batched Slab Matrix Math (`[Q, 384] @ [384, N]` up to `262,144` columns):**
   - Compares your M2's **CPU (Accelerate/BLAS)** vs. **Apple `MLX` Metal GPU** (and `PyTorch MPS` if installed) at `N = 4K, 16K, 64K, 262K` columns and batch sizes `Q = 1, 8, 16, 64`.
   - **The "aha" moment:** Scoring `16` concurrent agent steps or draft claims (`Q=16`) against **`262,144` vectors** takes only **a few milliseconds** locally — faster than a single TCP round-trip to a cloud API.
   - Notice how packing multiple queries together (`Q=8..64`) costs nearly the same wall time as `Q=1`. That's why packing an entire wave of subagents + claims into one matrix multiply scales effortlessly.

2. **Part B — Semantic Thrash & Silent-Stall Detection (`detect_thrash`):**
   - Runs the exact sliding-window math on two traces:
     - A **healthy 5-step debugging trace** (hits an error in the middle, introduces new files/symbols, applies fixes, ends with passing tests -> `thrash_score ≈ 0.0`, `action: continue`).
     - A **polite paraphrased stall loop** (zero error messages, polite phrasing, re-reading the same design notes 5 times with zero new code targets -> `thrash_score > 0.80`, `action: early_abort_and_reroute`).

3. **Part C — Zero-Token Claim Grounding Against Your Own Repo:**
   - Indexes up to 120 source files in your `--repo`, picks a real symbol from your repo, synthesizes a false agent claim (*"This repo is missing `<RealSymbol>`"*), and checks whether local embedding retrieval immediately flags the exact `file:line` proving the claim false — in milliseconds, at `$0.00` API cost.

The script writes a timestamped JSON receipt (`receipts/m2_math_receipt.json`) so you can compare backends after tuning models or quantization.

---

## Step 2: Watch Your Live Frontier Agents & Subagents (Claude, Codex, Hermes)

**Claude Code, Codex CLI, and Hermes are ALREADY writing append-only local JSONL transcripts to your Mac's SSD while they run:**

| Tool | Local Transcript Location on macOS | Includes Subagents? |
|---|---|---|
| **Claude Code** | `~/.claude/projects/<encoded-project-path>/<session-id>.jsonl` | **Yes!** Spawned subagents stream right next to it at `<session-id>/subagents/agent-<id>.jsonl` |
| **Codex CLI** | `~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl` | Yes (all tool calls, `apply_patch`, `update_plan`, shell traces) |
| **Hermes Agent** | `~/.hermes/logs/sessions/<timestamp>.jsonl` *(plus `~/.hermes/state.db`)* | Yes (reasoning turns, tool calls, and results) |

Your local watcher never intercepts API traffic and never touches your keys. It simply **tails those local JSONL files read-only** — and automatically filters out bare parallel tool acknowledgements (`"Created file."`, `"OK"`, `"Made 1 replacement..."`) so fast bursts of parallel edits don't cause false-positive loop alerts.

### 2A. Give Your Frontier Agents a Clean Starter Ruleset

Copy [`rules/frontier-agent-watchdog-skill.md`](rules/frontier-agent-watchdog-skill.md) into:
- **Claude Code:** `~/.claude/CLAUDE.md` (global) or your repo's `CLAUDE.md` (or `.claude/skills/watchdog-checkpoints/SKILL.md`)
- **Codex CLI:** your repo's `AGENTS.md`
- **Hermes:** your Hermes system/skill prompt folder

Why add this tiny ruleset?
1. It tells frontier agents and subagents to **state concrete target files and exit conditions up front**, and gives explicit permission to emit **`STUCK:`** instead of spiraling through polite re-reads.
2. Those crisp intent statements become the exact input vector (`intended_outcome`) your local watcher and JEPA dataset need.

### 2B. Run the Out-of-Band Watcher in a Side Terminal

In a separate terminal pane while Claude, Codex, or Hermes works:

```bash
# One-shot audit of your most recent active Claude / Codex / Hermes sessions:
python3 scripts/02_watch_frontier_agents.py --once

# Continuous live watch (polls every 5s, 0 API tokens, alerts on stalls/loops, logs episodes):
python3 scripts/02_watch_frontier_agents.py --poll-seconds 5 --episodes episodes/transitions.jsonl
```

Or target a specific transcript directly:

```bash
python3 scripts/02_watch_frontier_agents.py \
  --trace ~/.claude/projects/-Users-you-myrepo/abc123.jsonl \
  --intent "Add retry timeout guard to webhook_client.py and pass pytest"
```

---

## Step 3: Collect Local JEPA Training Data (Why & How)

### Why Not Just Save Raw Chat Logs?

Raw transcripts only tell you *what happened*. They don't tell a model **what surprised you**.

A **JEPA (Joint Embedding Predictive Architecture)** doesn't waste compute predicting raw English tokens like an LLM. Instead, it works purely in compact latent vector space (`384-D`):

$$\text{JEPA Predictor: } \quad f_\theta\big(\text{embed}(\text{current\_state}),\; \text{embed}(\text{intended\_outcome})\big) \;\longrightarrow\; \big(\widehat{\text{end\_state}},\; \hat{p}_{\text{stall/drift}}\big)$$

Every time your watcher observes a frontier agent finish (or stall on) a bounded task, it appends one **5-field Residual Episode** to `episodes/transitions.jsonl`:

1. **`current_state`** — Where the agent started.
2. **`intended_outcome`** — What you asked the agent or subagent to accomplish.
3. **`local_prediction`** — What your local watchdog predicted *before* seeing the outcome (predicted end embedding + predicted stall/drift risk).
4. **`end_state`** — What actually happened (files touched, final summary, observed `thrash_score`).
5. **`assessment_vs_expectations`** — The **prediction residual (`Δ`)**:
   - `state_surprise_delta = 1 - cosine(predicted_end_state, actual_end_state)`
   - `risk_prediction_error = |actual_risk - predicted_risk|`
   - Optional 1-line human tag (`"good"`, `"drifted_scope"`, `"hallucinated_missing_test"`, `"stall_loop"`).

### Why This Compounding Loop Gets Useful Fast

After **30–100 real agent runs** across Claude, Codex, and Hermes:
- **Early Risk Prediction (Before Spending Tokens):** Your tiny local JEPA learns which combinations of *prompt phrasing + repo area* historically trigger stall loops or scope drift — warning you *before* a subagent goes rogue.
- **High-Surprise Review Triage:** When you spawn 4 parallel subagents, your Mac ranks their outputs by **surprise (`Δ`)** so you inspect the weirdest / highest-risk diff first.
- **Personal Prompt & Rule Refinement:** Grouping your highest-surprise episodes immediately exposes recurring traps in your codebase that deserve a 1-line rule in `CLAUDE.md` / `AGENTS.md`.

### Train Your First Tiny Local JEPA on Your M2

Test the training loop today on synthetic Day-1 episodes (runs in `< 1 second` on your M2):

```bash
python3 scripts/03_train_tiny_jepa.py --demo-synthetic
```

Once `episodes/transitions.jsonl` collects `20+` real episodes from your Claude / Codex / Hermes runs, train on your own data:

```bash
python3 scripts/03_train_tiny_jepa.py \
  --episodes episodes/transitions.jsonl \
  --out checkpoints/my_m2_jepa.npz
```

Then feed `--jepa-checkpoint checkpoints/my_m2_jepa.npz` back into `02_watch_frontier_agents.py` so every future run gets live predictions from **your own locally trained JEPA** — and every new surprise sharpens the next round.

---

## Ideas to Hack On Next

- **Multi-subagent collision detector:** Pack the latest step embedding from every active subagent (`Q=8`) into one MatMul against your repo's file embeddings to spot two subagents silently touching the same module before git merge pain hits.
- **CoreML fixed-shape ANE experiment:** Export `all-MiniLM-L6-v2` with fixed static shape `[4, 128]` via `coremltools` and compare battery wattage (`powermetrics --samplers ane,gpu_power`) vs. `MLX` Metal GPU.
- **Pre-review quote check:** Before accepting a large agent diff, run its summary bullets through the claim-grounding pass in `01_see_the_math_m2.py` to verify cited files and symbols actually exist where the agent says.
- **Pair with NVIDIA's open-source [`NVIDIA/auto-ontology`](https://github.com/NVIDIA/auto-ontology) (Apache 2.0):** If your projects have messy schemas or overloaded domain terms, use `auto-ontology`'s human-in-the-loop workflow to certify definitions once, load those certified rules into your local `384-D` matrix so your M2 catches semantic drift at `$0` API tokens, and feed your highest-surprise Steppewatch episodes back into `auto-ontology` whenever an agent trips over an undocumented rule.
