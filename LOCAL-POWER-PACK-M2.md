# Steppewatch Local Power Pack (Apple M2 · 24 GB Unified Memory Edition)

> **Who this is for:** Builders on a **24 GB Apple Silicon Mac** (starting with the 2022 13" M2 MacBook Pro, `Mac14,7`, `100 GB/s` unified memory, 10-core Metal GPU, 16-core ANE) who already use cloud frontier agents (Claude Code, Codex CLI, Hermes) and want to know **what real local models actually shine on their desk** — grounded in memory-bandwidth physics, not hype.

---

## 1. The One Hardware Law Your M2 Already Proved

When Steppewatch's Option A benchmark ran on a 24 GB M2 MacBook Pro (`PyTorch MPS` Metal `FP16`, `192 MiB` matrix against `262,144` vectors):

- `Q = 1` query: **`8.43 ms`**
- `Q = 8` queries: **`10.14 ms`** *(8× more work for only **+1.71 ms**!)*
- `Q = 16` queries: **`10.20 ms`** *(**4.52× faster** than M2 CPU's `46.06 ms`)*
- `Q = 64` queries: **`13.71 ms`** *(**4.69× faster** than M2 CPU's `64.34 ms`)*

That single curve tells you **everything** about picking local models for a `100 GB/s` base M2:

1. **Why big dense chat models feel slow (`100 GB/s` bus):** Single-token autoregressive decoding has to stream the entire model's weights across memory for *every single output token* (`14B` Q4 ≈ `8–10 tok/s`; `24–32B` dense Q4 ≈ `4–5 tok/s`). **Let cloud frontier models write 1,500-line drafts.**
2. **Why attention-heavy verification & Multi-Token Prediction (MTP) / speculative decoding fly:**
   - **Bidirectional encoders & cross-attention rerankers** process whole chunks/diffs in parallel and output **0 or 1 token** — pure Metal FP16/FP32 compute (`~7.2 TFLOPS` FP16), zero decode bottleneck.
   - **MTP / speculative decoding** verifies `3–6` proposed tokens in **one single weight read** — the exact same `Q=1 -> Q=8` amortization your M2 just measured!
   - **Sparse MoEs (`~3B` active out of `30B–35B` total)** fit *only* because you have **24 GB unified memory** (16 GB Macs swap hard), while only reading `~3B` active weights per pass.

> **Apple Silicon Runtime Note (Important):** Prefer **`MLX`** (`mlx-lm` native `--draft-model` speculative decoding, or in-place MTP loaders like `mlx-optiq --mtp` / `MTPLX`) over `llama.cpp` for MTP/speculative decoding on Mac. On Apple Metal, `llama.cpp`'s speculative loop frequently breaks even or regresses due to draft verification overhead, and older MTP loaders double-loaded base weights.

---

## 2. Profile A — The Always-On Supervision Rig (`~5–6 GB` Total RAM)

**Best daily-driver stack.** Leaves **`18+ GB`** completely free for macOS, VS Code/Cursor, Chrome, Docker, and your remote Claude/Codex/Hermes sessions — zero swap, zero fan noise, `$0.00` extra API cost.

| Slot | Recommended Model(s) (MLX / MPS) | RAM | Role Alongside Frontier Agents |
|---|---|---:|---|
| **1. Long-Context Encoder** *(upgrades 256-tok MiniLM)* | **`Qwen3-Embedding-0.6B`** (`32K` ctx, MLX) or **`ModernBERT-large`** (`395M`, `8K` ctx) | `~0.8–1.5 GB` | Embeds whole files, full diffs, and multi-turn trajectories with real long-context attention instead of 1-sentence snippets. |
| **2. Cross-Attention Claim Verifier** *(the biggest accuracy jump)* | **`Qwen3-Reranker-0.6B`** (`<100 ms/batch`) or **`Qwen3-Reranker-4B`** (Q4 MLX) / **`BGE-Reranker-v2-M3`** | `~1.2–2.6 GB` | Bi-encoders struggle with negation (*"missing test X"* vs *"line 180 implements test X"*). A cross-encoder runs **full token-by-token cross-attention** over `(claim, retrieved source lines)` and outputs **1 entailment logit** — zero decode lag. |
| **3. Fast MTP / Speculative Triage LLM** | **Stock `mlx-lm` (turn-key today):** `Qwen3-8B` (4-bit MLX) + `Qwen3-0.6B` draft (`--draft-model`)<br>**Native MTP option:** `MiMo-7B` (Q4 `~4.5–5.5 GB`, pretrained from scratch with MTP) | `~4.5–5.5 GB` | Decodes at **`~15–28+ tok/s`**. Only wakes up when Steppewatch flags a stall or high-surprise episode (`Δ`) to explain *why* the cloud agent drifted and draft a 1-line `CLAUDE.md` / `AGENTS.md` guard. |

### Quick Test With Stock `mlx-lm` Speculative Decoding (macOS Apple Silicon)

```bash
pip install mlx-lm

# Example: battle-tested MLX draft+target speculative decode (~5.5 GB total RAM)
python3 -m mlx_lm generate \
  --model mlx-community/Qwen3-8B-4bit \
  --draft-model mlx-community/Qwen3-0.6B-4bit \
  --prompt "Summarize in 2 bullets why a silent paraphrased agent loop wastes tokens." \
  --max-tokens 120
```

*(Note: verify exact Hugging Face repo tags before pulling; always ask permission before downloading multi-GB weights on a user's machine.)*

---

## 3. Profile B — The 24 GB Solo Flex: Sparse MoE + Native MTP (`~17–20 GB` RAM)

If you want maximum local reasoning/coding IQ running **solo** (offline flight, private repo, or zero-API prototyping), **24 GB unified memory** unlocks the `30B–35B` Sparse MoE tier that 16 GB Macs cannot hold cleanly:

- **`Qwen3.5-35B-A3B` (4-bit MLX + in-place MTP, `35B` total / `~3B` active per token)** or **`Qwen3-30B-A3B` (4-bit MLX):**
  - **Why it works on base M2:** Total weights sit in unified memory (`~17–20 GB`), while every token pass only streams **`~3B` active parameters** across the `100 GB/s` bus (`~12–18+ tok/s`, faster with in-place MTP!).
  - **Honest 24 GB boundaries:**
    - Close heavy browser/Docker memory hogs before loading — leaves very little swap headroom.
    - Keep context moderate (`8K–16K`) so KV cache does not trigger macOS SSD swap.
    - Use **in-place MLX MTP** (never double-load weights). If macOS caps wired GPU memory tightly, inspect `sysctl iogpu.wired_limit_mb` before adjusting anything.

---

## 4. What NOT to Waste an Afternoon Trying on 24 GB Base M2

- **Giant `100B+` MTP MoEs (`Step-3.5-Flash 196B`, full `DeepSeek-V3/R1`):** Even though active parameters are small, total weights exceed `65–100+ GB` and physically cannot fit in 24 GB RAM.
- **Dense `24B–32B` models for interactive coding:** They fit in 24 GB, but at `~4–5 tok/s` on `100 GB/s` they feel sluggish compared to either Profile A's always-on supervision rig or Profile B's `35B-A3B` sparse MoE.

---

## 5. One-Paste Agent Prompt (Let Your Agent Check Your Mac First)

Paste this into Claude Code, Codex CLI, or Hermes to check your Mac's real free memory and MLX readiness **before** downloading a single byte of model weights:

```text
Read LOCAL-POWER-PACK-M2.md in steppewatch. Check my Mac's current RAM headroom (`sysctl hw.memsize` + `memory_pressure | tail -n 5`), check whether `mlx` and `mlx_lm` are installed in my active Python environment, and recommend either Profile A (Always-On ~5-6 GB Supervision Rig: encoder + cross-attention reranker + speculative/MTP triage) or Profile B (Solo 35B-A3B MoE). Do NOT download any models or change sysctl settings until I explicitly approve which model to pull.
```
