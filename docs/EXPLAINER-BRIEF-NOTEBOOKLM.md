# Steppewatch — Explainer Brief (NotebookLM Video / Audio Source)

> **Purpose of this document:** A clean, story-driven technical briefing designed specifically as a NotebookLM source file to generate a concise 3–5 minute explainer video or audio overview. Every number below is empirically verified from live hardware runs on November 21, 2026, with clear honesty about what is proven today vs. what comes next.  
> **Project:** Steppewatch (`github.com/djcdevelopment/steppewatch`) · **Author:** Derek Ciula (Steppe Integrations) · **License:** MIT

---

## Beat 1 — The Hook: The Supervision Paradox in AI Coding Agents

Developers everywhere are now running parallel cloud coding agents — Claude Code with spawned subagents, OpenAI Codex CLI, Hermes, and more. When those agents work well, they feel like magic.

When they go sideways, they fail in surprisingly expensive, subtle ways:
- **The Silent Polite Loop:** Instead of crashing loudly with an error, an agent politely paraphrases the same plan five turns in a row — *"Let me carefully review the integration architecture one more time before making edits"* — introducing zero new files, zero new symbols, and zero progress while quietly burning thousands of frontier tokens.
- **Confident False Claims:** Strong models routinely declare *"this repo is missing a test for X"* even when that exact test is sitting at line 180 of the file they just hashed — or they cite deprecated local ports and shift line numbers by 8 to 25 lines, forcing a human reviewer to spend 20 minutes manually re-deriving every anchor.

How do most teams try to fix runaway agents? **They hire a second expensive frontier LLM to watch the first one.** That creates an obvious trap: double the API spend, double the latency, and twice the rate-limit contention.

---

## Beat 2 — The Overlooked Asset Sitting Right Under Your Hands

While remote cloud agents are burning API tokens, open Activity Monitor or Task Manager on your laptop.

Almost every modern computer — an Apple Silicon M2/M3/M4 MacBook with a 16-core Neural Engine and 24 GB of unified memory, or a modern Windows/Linux laptop with an Intel or AMD NPU — has **dedicated matrix-math silicon sitting completely idle at 0% utilization.**

**Steppewatch** asks a simple systems question:
> *What if your idle local silicon watched your remote frontier agents in real time for `$0.00` in API tokens — AND turned every run into structured training data that makes your local watchdog smarter every week?*

---

## Beat 3 — How the Local Matrix Trick Works (Without Burning GPU VRAM)

Most developers think of local embeddings as tiny, chatty one-at-a-time lookups. Steppewatch flips that on its head by designing directly for how NPU and unified-memory matrix silicon actually wants to work:

- **One Pinned `96.5 MiB` Quantized Matrix (`262,144` columns × `384` dimensions, `INT8`):** That single tiny matrix — less than `0.4%` of a 24 GB Mac's RAM, and **`0.0 MiB`** of discrete GPU VRAM on an NPU workstation — is partitioned into clean logical namespaces:
  1. **Code & Doc Symbols** (your live workspace),
  2. **Certified Rules & Contracts** (project invariants and rules),
  3. **Live Agent Trajectories** (sliding windows of active agent steps), and
  4. **Historical Residual Episodes** (past runs and surprises).
- **One Batched Matrix Multiply Scores Everything at Once:** Pack `8`, `16`, or `64` live agent steps or draft claims into a single matrix multiply (`[Q, 384] @ [384, 262144]`). Because systolic matrix hardware amortizes overhead across batches, scoring an entire wave of parallel subagents against your whole codebase takes only **tens of milliseconds** — faster than a single network round-trip to a cloud API.
- **Zero Interception Required:** Tools like Claude Code (including its spawned subagents!), Codex CLI, and Hermes *already* write append-only local JSONL transcripts to your SSD while they work. Steppewatch simply tails those local files read-only.

---

## Beat 4 — Real Numbers From Live Hardware (Not Slideware)

On November 21, 2026, we tested this architecture on live Intel Arrow Lake NPU hardware (`OpenVINO INT8/FP16`, `0.0 MiB` discrete GPU VRAM, verified zero GPU device bindings) against a real multi-repo architectural draft that an independent frontier reviewer (`Claude Opus`) had just audited line-by-line:

1. **Massive Capacity & Speed:** Indexing `89` source and Markdown files (`848` symbols, zero unindexed files) took **`1.57 seconds`** warm (`17.65 ms` per file, `82.35%` shared-memory cache hits, `0` tensor rebinds) — using only **`0.92%`** of the `262,144`-column slab (`>99%` capacity still wide open!).
2. **Caught `3 out of 3` Real Substance Errors at `$0` API Cost:** In **`34 to 75 milliseconds`**, local zero-token claim grounding independently caught every single substance mistake the frontier reviewer found:
   - Caught the false *"missing test"* claim at **Rank #1** (pointing straight to line 180 where the test already shipped),
   - Caught a conflated protocol acknowledgement rule at **Rank #1**,
   - Caught a hallucinated deprecated local port at **Rank #2**, and
   - Flagged **`21 out of 26`** displaced line citations before a human ever had to look.
3. **Separated Productive Debugging From Polite Stall Loops:** In **`~27 milliseconds`**, sliding-window semantic thrash detection scored a real 5-step coding-and-fixing trace at **`0.0000` (`continue`)** — even though it hit an error in the middle — while flagging a polite 5-step paraphrased stall loop (zero errors, zero new code targets) at **`0.9278` (`early_abort_and_reroute`)**.
4. **Independent Verification on a 2022 Apple M2 MacBook Pro (24 GB Unified Memory):** On the same day, an independent builder ran Steppewatch's zero-download benchmark on an M2 MacBook Pro using its existing `PyTorch MPS` Metal backend (`192 MiB` FP16 matrix vs. `384 MiB` FP32 CPU against `262,144` vectors):
   - `Q = 1`: `8.43 ms` Metal vs. `8.83 ms` CPU
   - `Q = 8`: `10.14 ms` Metal vs. `34.75 ms` CPU (`3.4×` faster)
   - `Q = 16`: **`10.20 ms`** Metal vs. `46.06 ms` CPU (**`4.5×` faster** — scaling 16× in batch size added only **`+1.77 ms`**!)
   - `Q = 64`: **`13.71 ms`** Metal vs. `64.34 ms` CPU (**`4.7×` faster**)
   - Silent stall flagged at `0.8635` vs. `0.0000` healthy in **`2.31 ms`**, and local repo claim grounding ran in **`2.78 ms`**.
5. **Honest Real-World Calibration Lessons Caught Live:**
   - During a real parallel edit burst, terse tool acknowledgements (`"Created file."` ×3) initially caused a transient `0.72` false positive; filtering bare tool-ack stubs dropped it straight to **`0.0000`**.
   - Mixing raw per-line quote vectors into the same slab namespace as coarse symbol vectors crowded top-20 discovery — proving empirically that line anchors and symbol indices belong in separate namespaces.

---

## Beat 5 — Why JEPA Residuals Beat Passive Chat Logs

Raw chat transcripts only record *what happened*. They don't capture **what surprised you**.

Drawing on the **JEPA (Joint Embedding Predictive Architecture)** principle — predicting outcomes in compact latent vector space (`384-D`) rather than generating raw English text — Steppewatch logs a compact **5-field Prediction-Residual Episode** for every watched agent task:

1. **`current_state`** — Where the agent started.
2. **`intended_outcome`** — What the user or orchestrator asked it to do.
3. **`local_prediction`** — What the local watchdog predicted *before* seeing the result (predicted end vector + stall/drift risk).
4. **`end_state`** — What actually happened (files touched, thrash score, outcome).
5. **`assessment_vs_expectations`** — The **surprise delta (`Δ`)** between prediction and reality.

**Why this compounds:**
- Real-time encoding, slab scoring, and surprise harvesting run on the NPU / Apple Silicon at `0` discrete GPU VRAM and `$0` API tokens.
- A tiny local JEPA MLP predictor (`< 2M` parameters) retrains in milliseconds to seconds on local CPU or unified memory, learning which prompts and repository areas tend to cause stalls or scope drift — so your local lookout gets sharper every week.
- *(Honest boundary: real-time NPU supervision, claim grounding, and residual harvesting are live-verified today; multi-week training curves over hundreds of real production episodes are the active next frontier.)*

---

## Beat 6 — Open Ecosystem: Pairing With NVIDIA `auto-ontology` & Self-Tapping Agents

Steppewatch is built to play cleanly with the open-source ecosystem:
- **Pairs with NVIDIA's Apache-2.0 `NVIDIA/auto-ontology`:** `auto-ontology` excels at discovering domain terminology across schemas/docs and putting human reviewers in the loop to certify canonical rules. Steppewatch compiles those certified rules straight into its zero-token local `NS_ONTOLOGY` slab — and feeds high-surprise residual agent failures right back into `auto-ontology`'s human certification queue.
- **Agents Can Self-Tap:** With `AGENTS.md` and a single `scripts/self_tap.py` JSON probe, Claude Code, Codex CLI, or Hermes can verify the local silicon, adopt a clean `INTENT:` / `STUCK:` / `END_STATE:` checkpoint contract, and check their own subagents for loops automatically.
- **Try It in 5 Minutes on Your Own Hardware:** Clone `github.com/djcdevelopment/steppewatch`, run `scripts/01_see_the_math_m2.py` against any local repo with zero API keys, and see the real millisecond numbers on your own machine.
