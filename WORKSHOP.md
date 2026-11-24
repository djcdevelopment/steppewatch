# Steppewatch Self-Discovery JEPA Workshop (15 Minutes, `$0.00` API Cost)

> **What this is:** A transparent, hands-on starter workshop for builders who want to experience what their own local silicon (Apple Silicon Metal/ANE or Intel/AMD NPU) can do alongside remote frontier agents — using **your own real Claude Code, Codex CLI, or Hermes transcripts**, not toy examples.

---

## Stage 0 — Honest Starting Line (Hello World vs. The Real Workshop)

- **Running `01_see_the_math_m2.py --force-hash` is "Hello World."** It proves your local hardware's batched matrix math speed (for example, `10.20 ms` for `Q=16` against `262,144` vectors on a 24 GB M2 MacBook Pro via PyTorch MPS) — **but** `--force-hash` uses zero-download lexical word/trigram hashing, *not* real semantic embeddings yet.
- **The real workshop starts right here:** pointing `04_xray_my_sessions.py` and `03_train_tiny_jepa.py` at **your own existing local agent history**, auditing whether flagged loops are genuine or false positives, comparing lexical hashing against real `all-MiniLM-L6-v2` embeddings on your hardware, and inspecting what surprises your first tiny local JEPA predictor.

Everything stays **100% read-only on your agent transcripts** and costs **`$0.00`** in extra cloud tokens.

---

## Fast Track: Hand This Prompt Straight to Your Agent

If you live inside **Claude Code**, **Codex CLI**, or **Hermes**, let your agent do the heavy lifting and walk you through all 4 discoveries interactively. Copy and paste this prompt as-is:

```text
Pull the latest https://github.com/djcdevelopment/steppewatch.git (v1.5+) and run a guided, read-only local-silicon JEPA workshop with me on my machine (do not modify my global configs or logs):

1. REAL SESSION X-RAY: Run `python3 scripts/04_xray_my_sessions.py --force-hash --max-sessions 25 --export-episodes episodes/my_history.jsonl --json` against my real local Claude Code / Codex / Hermes transcripts. Show me the top 3 sessions by thrash or intent-drift, inspect the #1 flagged session directly, and tell me honestly: did my agent actually stall or drift there, or was it a false positive?
2. TRAIN MY FIRST REAL JEPA: Run `python3 scripts/03_train_tiny_jepa.py --force-hash --episodes episodes/my_history.jsonl --out checkpoints/my_real_jepa.npz`. Show me the before/after validation state_surprise and risk_mae, and explain in plain English what kinds of prompts or repos caused my biggest surprises.
3. LEXICAL HASH vs REAL SEMANTIC EMBEDDINGS: Ask my permission before installing anything — then explain what changes in stall detection and claim grounding if we install `sentence-transformers` (~90 MB `all-MiniLM-L6-v2`) on my local accelerator (Apple MPS / MLX / NPU / CPU) and re-run steps 1–2 as a clean A/B comparison.
4. BUILDER CHALLENGE: After showing me my real results, propose 2 concrete upgrades we could hack into `scripts/03_train_tiny_jepa.py` or `scripts/02_watch_frontier_agents.py` (for example: cosine/EMA JEPA loss, multi-subagent file-collision alerts, or personal repo guardrails) and ask me which one I want to build first.
```

---

## Driving It Yourself (4 Hands-On Checkpoints)

Prefer running every step in your own terminal? Follow these four stages and answer the discovery question at each checkpoint:

### Discovery 1 — X-Ray Your Own Recent Agent History (Fast Hash Baseline)
```bash
git pull
python3 scripts/04_xray_my_sessions.py \
  --force-hash \
  --max-sessions 25 \
  --export-episodes episodes/my_history.jsonl
```
**Ask yourself:** Open the `#1` advisory-flagged transcript and look at the flag neighborhood (`flag@N` preview) and later outcomes.
- Did the agent actually enter a repeated-failure or no-progress stall, or was it productive inspection, sequential reading, or an intentional wait? (Keep flags **advisory**, not automatic termination.)
- Note: `post_flag_est_tokens` is a rough character-derived step-summary count (`chars / 4`) including unverified flags — **not** API billing or proven token savings.

### Discovery 2 — Train Your First Personal Tiny JEPA on Real History (`< 1 s` NumPy Train)
```bash
python3 scripts/03_train_tiny_jepa.py \
  --force-hash \
  --episodes episodes/my_history.jsonl \
  --out checkpoints/my_real_jepa.npz
```
**Ask yourself:** Open [`scripts/03_train_tiny_jepa.py`](scripts/03_train_tiny_jepa.py) (plain NumPy — zero magic black boxes).
- How much did chronological validation `state_surprise` (`1 - cos(ŝ_next, s_next)`) and `risk_mae` change from before to after training?
- Remember that until you populate `human_label` (`"stall"` / `"ok"`) in `episodes/my_history.jsonl`, risk targets come from the heuristic detector proxy rather than adjudicated ground truth.
- Look at your highest-surprise episodes in `episodes/my_history.jsonl`: did requirements change mid-session, or was there an undocumented repo rule that belongs as a 1-line guard in `CLAUDE.md` / `AGENTS.md`?

### Discovery 3 — Upgrade From Lexical Hashing to Real Semantic Embeddings (A/B Test)
Once you've seen the zero-download `--force-hash` baseline, install `sentence-transformers` (downloads `all-MiniLM-L6-v2` ~90 MB once, caches locally, and automatically runs on PyTorch MPS Metal on Apple Silicon; X-Ray encodes each session in a single batched pass, though total wall time scales with session count and hardware):

```bash
pip install -r requirements-m2.txt
python3 scripts/04_xray_my_sessions.py --max-sessions 25 --export-episodes episodes/my_history_minilm.jsonl
python3 scripts/03_train_tiny_jepa.py --episodes episodes/my_history_minilm.jsonl --out checkpoints/my_minilm_jepa.npz
```
**Ask yourself:** Compare the `--force-hash` run against `all-MiniLM-L6-v2`:
- Which paraphrased loops did MiniLM catch that lexical hashing missed?
- Which false-positive flags disappeared once synonyms and code semantics were understood?

### Discovery 4 — Make It Yours (3 Weekend Builder Challenges)

`Steppewatch` intentionally keeps its starter JEPA predictor and watcher small and readable so you learn by building:

1. **Upgrade the JEPA Loss & Architecture (`scripts/03_train_tiny_jepa.py`):** Swap Euclidean MSE for direct cosine-similarity loss, add an exponential moving average (EMA) target projection, or split predictions by agent harness (`claude` vs. `codex` vs. `hermes`).
2. **Multi-Subagent File-Collision Radar (`scripts/02_watch_frontier_agents.py`):** When 2+ subagents are active in parallel, pack their latest step vectors (`Q = 8`) into one batched Metal/NPU MatMul against your repo's file symbols — alert immediately if two subagents converge on the same module before a merge conflict lands.
3. **Personal Certified Rule Guardrails (Optional [`NVIDIA/auto-ontology`](https://github.com/NVIDIA/auto-ontology) Interop):** Write a tiny JSON list of 5–10 "never violate these rules in my repo" invariants (or export certified definitions from `NVIDIA/auto-ontology`), embed them locally, and score every agent's final summary against them in milliseconds at `$0.00` token cost.
4. **Upgrade Your Silicon Stack ([`LOCAL-POWER-PACK-M2.md`](LOCAL-POWER-PACK-M2.md)):** Pair a long-context encoder (`Qwen3-Embedding-0.6B` / `ModernBERT`) with a **cross-attention claim verifier** (`Qwen3-Reranker-0.6B/4B`) and a fast `mlx-lm` speculative/MTP triage model (`MiMo-7B` or `Qwen3-8B + 0.6B draft`) — all resident in `~5–6 GB` of a 24 GB Mac — or flex solo with `Qwen3.5-35B-A3B` sparse MoE.
