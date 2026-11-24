#!/usr/bin/env python3
"""Step 1 — See the Math on Your Own Machine (Apple M2 / Client Silicon).

Zero API keys and zero cloud tokens required. Demonstrates:
  Part A: Quantized INT8/FP16 multi-namespace slab MatMul latency [Q, 384] @ [384, N]
          up to N = 262,144 columns (~96.5 MiB INT8) on CPU vs. Apple MLX / PyTorch MPS.
  Part B: Sliding-window semantic thrash detection separating a productive debugging
          trace from a polite paraphrased stall loop.
  Part C: Zero-token claim grounding against a real local code repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from local_embedder import (
    EMBED_DIM,
    LocalEmbedder,
    detect_agent_thrash,
    probe_hardware,
)

CODE_EXTS = {".py", ".cs", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".md"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "bin", "obj", "dist", "build", "__pycache__"}
SYMBOL_DEF_RE = re.compile(
    r"^\s*(?:def|async\s+def|class|function|export\s+(?:async\s+)?function|func|fn|"
    r"(?:public|private|internal|protected)\s+(?:static\s+|async\s+|sealed\s+|partial\s+)*"
    r"(?:class|record|struct|interface|[A-Za-z0-9_<>\[\]?]+)\s+)\s*([A-Za-z_][A-Za-z0-9_]{3,})"
)


def bench_matmul_backends(
    cols_list: Tuple[int, ...] = (4096, 16384, 65536, 262144),
    q_list: Tuple[int, ...] = (1, 8, 16, 64),
    iters: int = 8,
) -> Dict[str, Any]:
    """Benchmark [Q, 384] @ [384, N] across installed local backends."""
    rng = np.random.default_rng(42)
    max_n = max(cols_list)
    # Build INT8 slab + per-column FP16 scales (exact ~96.5 MiB footprint at 262,144 cols)
    w_fp32 = rng.standard_normal((EMBED_DIM, max_n), dtype=np.float32)
    col_scales = np.maximum(np.max(np.abs(w_fp32), axis=0, keepdims=True) / 127.0, 1e-6).astype(np.float16)
    w_int8 = np.clip(np.round(w_fp32 / col_scales.astype(np.float32)), -127, 127).astype(np.int8)
    w_fp16 = w_fp32.astype(np.float16)
    slab_int8_mib = round((w_int8.nbytes + col_scales.nbytes) / (1024 * 1024), 2)
    slab_fp16_mib = round(w_fp16.nbytes / (1024 * 1024), 2)

    results: Dict[str, Any] = {
        "max_columns": max_n,
        "embed_dim": EMBED_DIM,
        "slab_int8_plus_scales_mib": slab_int8_mib,
        "slab_fp16_mib": slab_fp16_mib,
        "cpu_fp32_ms": {},
    }

    # 1. CPU float32 baseline (Accelerate BLAS on macOS / OpenBLAS on Linux)
    for n in cols_list:
        w_view = w_fp32[:, :n]
        results["cpu_fp32_ms"][str(n)] = {}
        for q in q_list:
            q_mat = rng.standard_normal((q, EMBED_DIM), dtype=np.float32)
            _ = q_mat @ w_view  # warmup
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = q_mat @ w_view
                times.append((time.perf_counter() - t0) * 1000.0)
            results["cpu_fp32_ms"][str(n)][f"Q={q}"] = round(float(np.median(times)), 3)

    # 2. Apple Silicon MLX Metal GPU (Unified Memory, zero PCIe copy)
    try:
        import mlx.core as mx  # type: ignore

        results["mlx_metal_fp16_ms"] = {}
        w_mx_full = mx.array(w_fp16)
        mx.eval(w_mx_full)
        for n in cols_list:
            w_mx = w_mx_full[:, :n]
            results["mlx_metal_fp16_ms"][str(n)] = {}
            for q in q_list:
                q_mx = mx.array(rng.standard_normal((q, EMBED_DIM), dtype=np.float32).astype(np.float16))
                out = mx.matmul(q_mx, w_mx)
                mx.eval(out)  # warmup
                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = mx.matmul(q_mx, w_mx)
                    mx.eval(out)
                    times.append((time.perf_counter() - t0) * 1000.0)
                results["mlx_metal_fp16_ms"][str(n)][f"Q={q}"] = round(float(np.median(times)), 3)
    except Exception:
        pass

    # 3. PyTorch MPS Metal GPU fallback (if torch MPS is installed on Mac)
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            results["torch_mps_fp16_ms"] = {}
            w_mps_full = torch.from_numpy(w_fp16).to("mps")
            torch.mps.synchronize()
            for n in cols_list:
                w_mps = w_mps_full[:, :n]
                results["torch_mps_fp16_ms"][str(n)] = {}
                for q in q_list:
                    q_mps = torch.randn((q, EMBED_DIM), dtype=torch.float16, device="mps")
                    _ = q_mps @ w_mps
                    torch.mps.synchronize()
                    times = []
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        _ = q_mps @ w_mps
                        torch.mps.synchronize()
                        times.append((time.perf_counter() - t0) * 1000.0)
                    results["torch_mps_fp16_ms"][str(n)][f"Q={q}"] = round(float(np.median(times)), 3)
    except Exception:
        pass

    return results


def run_thrash_demo(embedder: LocalEmbedder) -> Dict[str, Any]:
    """Compare a productive debugging trace against a polite paraphrased stall loop."""
    healthy_steps = [
        "Step 1: Inspected webhook_client.py send_event() to verify current retry backoff and timeout defaults.",
        "Step 2: Edited webhook_client.py to add max_timeout_sec=15.0 and fixed TypeError on RetryConfig init.",
        "Step 3: Updated test_webhook_client.py to add test_retry_timeout_guard covering HTTP 504 responses.",
        "Step 4: Ran pytest test_webhook_client.py; caught AssertionError at line 88 and patched backoff jitter.",
        "Step 5: Re-ran pytest test_webhook_client.py; all 12 tests passed with exit code 0.",
    ]
    polite_stall_steps = [
        "Step 1: Let me carefully review the webhook integration requirements again to make sure I have the full picture before proposing an edit.",
        "Step 2: I will re-examine the webhook integration plan and verify how the retry timeout should be structured before making any code edits.",
        "Step 3: Let me double-check the webhook integration architecture one more time so the proposed design aligns with existing client boundaries.",
        "Step 4: Before writing any code, I want to carefully reconsider the webhook integration approach and review the timeout constraints again.",
        "Step 5: I am revisiting the webhook integration design notes again to confirm the retry timeout details before deciding on the next edit.",
    ]
    return {
        "healthy_trace": detect_agent_thrash(embedder, healthy_steps, trace_id="demo-healthy"),
        "polite_stall_loop": detect_agent_thrash(embedder, polite_stall_steps, trace_id="demo-polite-stall"),
    }


def _lexical_symbol_scores(entries: List[Dict[str, Any]], query: str) -> np.ndarray:
    """Fast exact-symbol and path lexical matcher (complements dense embeddings)."""
    q_low = (query or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9_./:-]{2,}", q_low))
    q_words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", q_low))
    scores = np.zeros((len(entries),), dtype=np.float32)
    if not q_tokens:
        return scores

    for idx, entry in enumerate(entries):
        sym = str(entry.get("symbol", "")).lower()
        rel = str(entry.get("rel", "")).lower()
        txt_low = str(entry.get("text", "")).lower()
        e_tokens = set(re.findall(r"[a-z0-9_./:-]{2,}", txt_low))
        overlap = len(q_tokens & e_tokens) / float(max(1, len(q_tokens)))
        sym_hit = 1.0 if (sym and (sym in q_words or re.search(rf"\b{re.escape(sym)}\b", q_low))) else 0.0
        rel_hit = 1.0 if (rel and rel in q_low) else (0.5 if (rel and Path(rel).name in q_low) else 0.0)
        scores[idx] = float(0.45 * sym_hit + 0.30 * rel_hit + 0.25 * overlap)
    return scores


def _hybrid_rrf_scores(dense_scores: np.ndarray, lexical_scores: np.ndarray, k: int = 20) -> np.ndarray:
    """Combine dense and lexical rankings via Reciprocal Rank Fusion + exact lexical bonus."""
    n = len(dense_scores)
    dense_order = np.argsort(-dense_scores)
    lex_order = np.argsort(-lexical_scores)
    rrf = np.zeros((n,), dtype=np.float32)
    for rank, idx in enumerate(dense_order):
        rrf[int(idx)] += 1.0 / float(k + rank + 1)
    if float(np.max(lexical_scores)) > 0.0:
        for rank, idx in enumerate(lex_order):
            if lexical_scores[int(idx)] > 0.0:
                rrf[int(idx)] += 1.0 / float(k + rank + 1)
        # Preserve strong exact-symbol + exact-file matches when dense tokenization splits identifiers.
        rrf += 0.04 * lexical_scores
    return rrf


def run_repo_claim_grounding(
    embedder: LocalEmbedder,
    repo_dir: Path,
    max_files: int = 120,
    use_rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    offline: bool = False,
) -> Dict[str, Any]:
    """Index local files and surface candidate source evidence for a synthetic claim."""
    t0 = time.perf_counter()
    entries: List[Dict[str, Any]] = []
    discovered_symbols: List[Tuple[str, int, str, str]] = []
    file_count = 0

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for fname in sorted(files):
            fp = Path(root) / fname
            if fp.suffix.lower() not in CODE_EXTS or fp.is_symlink():
                continue
            try:
                if fp.stat().st_size > 120_000:
                    continue
                lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            file_count += 1
            rel = str(fp.relative_to(repo_dir))
            for ln_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or len(stripped) < 8:
                    continue
                if fp.suffix.lower() == ".md" and stripped.startswith("#"):
                    entries.append({"rel": rel, "line": ln_no, "symbol": "", "text": f"{rel}:{ln_no} | {stripped}"})
                else:
                    m = SYMBOL_DEF_RE.match(line)
                    if m:
                        sym = m.group(1)
                        entries.append({"rel": rel, "line": ln_no, "symbol": sym, "text": f"{rel}:{ln_no} | {stripped}"})
                        if sym not in {"main", "init", "setup", "test"} and len(discovered_symbols) < 40:
                            discovered_symbols.append((rel, ln_no, sym, stripped))
            if file_count >= max_files:
                break
        if file_count >= max_files:
            break

    if not entries:
        return {"skipped": True, "reason": f"No indexable code/doc symbols found under {repo_dir}"}

    texts = [e["text"] for e in entries]
    slab = embedder.encode(texts)  # [M, 384]
    index_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Pick a real symbol (or top heading) and synthesize a false agent claim
    if discovered_symbols:
        rel, ln_no, sym, raw_line = discovered_symbols[len(discovered_symbols) // 2]
        false_claim = f"{rel} is missing an implementation or test for {sym}"
    else:
        e0 = entries[0]
        rel, ln_no, sym, raw_line = e0["rel"], e0["line"], e0["text"], e0["text"]
        false_claim = f"{rel} does not document {sym}"

    t_q = time.perf_counter()
    q_vec = embedder.encode([false_claim])[0]
    dense_scores = slab @ q_vec
    lexical_scores = _lexical_symbol_scores(entries, false_claim)
    hybrid_scores = _hybrid_rrf_scores(dense_scores, lexical_scores)

    dense_order = np.argsort(-dense_scores)
    lex_order = np.argsort(-lexical_scores)
    hybrid_order = np.argsort(-hybrid_scores)
    query_ms = round((time.perf_counter() - t_q) * 1000.0, 2)

    exact_ref = f"{rel}:{ln_no}"

    def _find_rank(order: np.ndarray) -> int | None:
        for r_i, idx in enumerate(order, 1):
            if f"{entries[int(idx)]['rel']}:{entries[int(idx)]['line']}" == exact_ref:
                return r_i
        return None

    dense_rank = _find_rank(dense_order)
    lexical_rank = _find_rank(lex_order)
    hybrid_rank = _find_rank(hybrid_order)

    active_order = hybrid_order
    active_scores = hybrid_scores
    rerank_info: Dict[str, Any] = {"enabled": False}

    if use_rerank:
        pool_idx = [int(i) for i in hybrid_order[:20]]
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            t_r0 = time.perf_counter()
            local_only = offline or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            ce_kwargs: Dict[str, Any] = {"local_files_only": True} if local_only else {}
            ce = CrossEncoder(rerank_model, **ce_kwargs)
            pairs = [(false_claim, entries[i]["text"]) for i in pool_idx]
            ce_scores = np.asarray(ce.predict(pairs), dtype=np.float32)
            rerank_ms = round((time.perf_counter() - t_r0) * 1000.0, 2)
            local_sort = np.argsort(-ce_scores)
            reranked_pool = [pool_idx[int(j)] for j in local_sort]
            active_order = np.array(reranked_pool, dtype=np.int64)
            active_scores = np.zeros_like(hybrid_scores)
            for j, idx in enumerate(pool_idx):
                active_scores[idx] = float(ce_scores[j])
            rerank_rank = _find_rank(active_order)
            rerank_info = {
                "enabled": True,
                "model": rerank_model,
                "pool_size": len(pool_idx),
                "rerank_ms": rerank_ms,
                "rerank_rank": rerank_rank,
                "note": "Cross-encoder scores query/code relevance for human inspection, not logical entailment.",
            }
        except Exception as exc:
            rerank_info = {"enabled": False, "requested": True, "error": str(exc)}

    top_idx = active_order[:5]
    top_hits = [
        {
            "rank": rank + 1,
            "ref": f"{entries[int(i)]['rel']}:{entries[int(i)]['line']}",
            "score": round(float(active_scores[int(i)]), 4),
            "dense_score": round(float(dense_scores[int(i)]), 4),
            "lexical_score": round(float(lexical_scores[int(i)]), 4),
            "line_text": entries[int(i)]["text"][:140],
        }
        for rank, i in enumerate(top_idx)
    ]
    caught_rank = next((h["rank"] for h in top_hits if h["ref"] == exact_ref), None)

    return {
        "repo": str(repo_dir),
        "files_scanned": file_count,
        "symbols_indexed": len(entries),
        "index_ms": index_ms,
        "query_ms": query_ms,
        "retrieval_mode": "hybrid_rrf_plus_rerank" if rerank_info.get("enabled") else "hybrid_lexical_plus_dense",
        "grounding_scope": "candidate_evidence_for_inspection_not_entailment",
        "synthetic_false_claim": false_claim,
        "ground_truth_ref": exact_ref,
        "dense_rank": dense_rank,
        "lexical_rank": lexical_rank,
        "hybrid_rank": hybrid_rank,
        "caught_in_top5": caught_rank is not None,
        "caught_rank": caught_rank,
        "rerank": rerank_info,
        "top5_hits": top_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="See the local silicon math for agent supervision.")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Local repo directory to index for Part C")
    parser.add_argument("--force-hash", action="store_true", help="Skip SentenceTransformer download; use fast hash384")
    parser.add_argument("--offline", action="store_true", help="Load local models from cache only; forbid Hugging Face network calls")
    parser.add_argument("--rerank", action="store_true", help="Optional bounded CrossEncoder reranking over top-20 hybrid candidates")
    parser.add_argument(
        "--rerank-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Optional CrossEncoder model for --rerank",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("receipts/m2_math_receipt.json"),
        help="Path to write JSON benchmark receipt",
    )
    args = parser.parse_args()

    repo_dir = args.repo.expanduser().resolve()
    if not repo_dir.is_dir():
        print(f"Error: --repo path does not exist or is not a directory: {args.repo}", file=sys.stderr)
        return 1

    hw = probe_hardware()
    embedder = LocalEmbedder(force_hash=args.force_hash, offline=args.offline)

    print("=== Hardware & Local Backends ===")
    print(json.dumps(hw, indent=2))
    print(f"Active 384-D Embedder: {embedder.backend_name}")
    if embedder.fallback_reason:
        print(f"[note] {embedder.fallback_reason}")
    print()

    print("=== Part A: Random-Matrix Slab Microbenchmark ([Q, 384] @ [384, N]) ===")
    mm = bench_matmul_backends()
    print(
        f"Pinned 262,144-column slab footprint: {mm['slab_int8_plus_scales_mib']} MiB (INT8+FP16 scales reference) "
        f"| {mm['slab_fp16_mib']} MiB (FP16 timed on MLX/MPS)"
    )
    for backend_key in ("cpu_fp32_ms", "mlx_metal_fp16_ms", "torch_mps_fp16_ms"):
        if backend_key not in mm:
            continue
        print(f"\n  Backend: {backend_key}")
        for n_str, q_map in mm[backend_key].items():
            row = "  ".join(f"{k}: {v:>6.2f} ms" for k, v in q_map.items())
            print(f"    N={int(n_str):>6d} cols -> {row}")

    print("\n=== Part B: Synthetic 2-Trace Loop Smoke Check (Advisory Scoring) ===")
    thrash = run_thrash_demo(embedder)
    for name, res in thrash.items():
        print(
            f"  {name:20s} -> detected={str(res['thrash_detected']):5s} "
            f"score={res['thrash_score']:.4f} action={res['action']} "
            f"(adj_cos={res['mean_adj_cosine']}, stag={res['stagnation_ratio']}, "
            f"new_targets={res['tail_new_targets']}, {res['latency_ms']} ms)"
        )

    print(f"\n=== Part C: Hybrid Zero-Token Candidate Grounding on {repo_dir} ===")
    grounding = run_repo_claim_grounding(
        embedder,
        repo_dir,
        use_rerank=args.rerank,
        rerank_model=args.rerank_model,
        offline=args.offline,
    )
    if grounding.get("skipped"):
        print(f"  Skipped: {grounding['reason']}")
    else:
        print(
            f"  Indexed {grounding['symbols_indexed']} symbols across {grounding['files_scanned']} files "
            f"in {grounding['index_ms']} ms; hybrid query took {grounding['query_ms']} ms"
        )
        print(f"  Synthetic claim : {grounding['synthetic_false_claim']}")
        print(
            f"  Ground truth    : {grounding['ground_truth_ref']} -> "
            f"hybrid_rank={grounding['hybrid_rank']} (lexical={grounding['lexical_rank']}, dense={grounding['dense_rank']})"
        )
        if grounding["rerank"].get("enabled"):
            print(
                f"  Cross-encoder   : rank={grounding['rerank']['rerank_rank']} "
                f"in {grounding['rerank']['rerank_ms']} ms (relevance only, not entailment)"
            )
        for h in grounding["top5_hits"][:3]:
            print(f"    #{h['rank']} ({h['score']:.4f}) {h['ref']}  {h['line_text'][:90]}")

    receipt = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw,
        "embedder": embedder.backend_name,
        "part_a_slab_matmul": mm,
        "part_b_thrash_detection": thrash,
        "part_c_repo_claim_grounding": grounding,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    fastest_backend = "mlx_metal_fp16_ms" if "mlx_metal_fp16_ms" in mm else (
        "torch_mps_fp16_ms" if "torch_mps_fp16_ms" in mm else "cpu_fp32_ms"
    )
    timed_precision = "FP16" if fastest_backend != "cpu_fp32_ms" else "FP32"
    timed_mib = mm["slab_fp16_mib"] if timed_precision == "FP16" else round(mm["slab_fp16_mib"] * 2.0, 2)
    q16_262k = mm[fastest_backend]["262144"]["Q=16"]
    h_score = thrash["healthy_trace"]["thrash_score"]
    s_score = thrash["polite_stall_loop"]["thrash_score"]
    print("\n=== Local Silicon Microbenchmark & Smoke-Check Summary ($0.00 API Cost) ===")
    print(
        f"  1. Matrix math  : Random-matrix microbenchmark Q=16 vs 262,144 cols ({timed_mib} MiB {timed_precision} timed; "
        f"{mm['slab_int8_plus_scales_mib']} MiB INT8+scales reference) in {q16_262k:.2f} ms ({fastest_backend})."
    )
    print(
        f"  2. Smoke check  : Separated 2 synthetic traces — healthy ({h_score:.4f}) vs polite stall ({s_score:.4f}); "
        "real-session flags remain advisory."
    )
    if not grounding.get("skipped"):
        print(
            f"  3. Candidate ref: Indexed {grounding['symbols_indexed']} symbols in {grounding['index_ms']} ms; "
            f"surfaced ground-truth symbol at Rank #{grounding['caught_rank']} for inspection ({grounding['query_ms']} ms)."
        )
    print(f"\nSaved receipt -> {args.receipt}")
    print("Next step (read-only on your own Claude Code / Codex / Hermes logs):")
    print("  python3 scripts/02_watch_frontier_agents.py --once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
