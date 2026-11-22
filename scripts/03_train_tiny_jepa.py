#!/usr/bin/env python3
"""Step 3 — Train a Tiny Local JEPA Transition & Risk Predictor on Your Machine.

Learns directly in 384-D latent space from 5-field Residual Episodes:
    f_theta(z_current_state, z_intended_outcome) -> (z_hat_end_state, p_hat_risk)

Zero cloud API calls. Pure NumPy backprop (runs in <2 seconds on Apple M2 CPU
for hundreds of episodes, zero framework friction) and saves a portable `.npz`
checkpoint that `02_watch_frontier_agents.py --jepa-checkpoint` loads live.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from local_embedder import EMBED_DIM, LocalEmbedder


def build_demo_synthetic_episodes(n_pairs: int = 36) -> List[Dict[str, Any]]:
    """Generate realistic healthy vs. stalled/drifted agent episodes for Day-1 testing."""
    templates = [
        # Healthy bounded engineering tasks -> low risk, end_state aligns with intent
        (
            "Inspecting auth_middleware.py token refresh logic on branch feat/auth",
            "Add expiry jitter check in auth_middleware.py and pass pytest",
            "Edited auth_middleware.py, added test_expiry_jitter in test_auth.py; 14 tests passed exit code 0",
            0.05,
        ),
        (
            "Inspecting webhook_client.py HTTP retry handler",
            "Add 15s timeout guard and HTTP 504 backoff in webhook_client.py",
            "Updated webhook_client.py with max_timeout_sec=15; test_webhook_client.py passed 12/12",
            0.04,
        ),
        (
            "Reading parser.ts AST span validator",
            "Fix off-by-one line number calculation in parser.ts and run npm test",
            "Patched parser.ts line offset math; added regression spec; npm test passed 0 errors",
            0.06,
        ),
        (
            "Reviewing sqlite_store.py schema migration script",
            "Add idempotent column migration for episodes table and verify unit tests",
            "Added ALTER TABLE guard in sqlite_store.py; pytest test_store.py passed 9/9",
            0.05,
        ),
        # Vague/unbounded prompts -> high stall or scope-drift risk
        (
            "Looking across the entire repository trying to understand all services",
            "Carefully review everything and make the whole architecture cleaner and better",
            "Re-reading architecture notes again to reconsider the full plan before deciding on any edit",
            0.92,
        ),
        (
            "Scanning multiple config and deployment directories without target files",
            "Investigate why things feel slow and rethink the whole system design",
            "Revisiting design constraints one more time to verify every subsystem before touching code",
            0.89,
        ),
        (
            "Inspecting legacy API adapter with conflicting specs",
            "Rewrite the communications layer without breaking anything anywhere",
            "Double-checking integration notes again; still reviewing boundaries with zero files modified",
            0.94,
        ),
        (
            "Reading old test logs and unrelated docs",
            "Fix all possible edge cases across every module in the repo",
            "Re-examining requirements again before making edits; no new targets or test runs yet",
            0.91,
        ),
    ]
    episodes: List[Dict[str, Any]] = []
    for i in range(n_pairs):
        curr, intent, end, risk = templates[i % len(templates)]
        suffix = f" [run-{i + 1}]"
        episodes.append(
            {
                "episode_id": f"demo-ep-{i + 1:03d}",
                "current_state": {"summary": curr + suffix},
                "intended_outcome": {"summary": intent + suffix},
                "end_state": {"summary": end + suffix},
                "assessment_vs_expectations": {"actual_risk_target": risk},
            }
        )
    return episodes


def load_episodes(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "current_state" in obj and "intended_outcome" in obj and "end_state" in obj:
                episodes.append(obj)
        except Exception:
            continue
    return episodes


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def train_tiny_jepa(
    X: np.ndarray,
    Y_state: np.ndarray,
    Y_risk: np.ndarray,
    hidden_dim: int = 192,
    epochs: int = 200,
    lr: float = 0.20,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Train a 2-layer latent JEPA predictor (768 -> hidden -> 384 state + 1 risk)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    tr_idx, va_idx = perm[:split], perm[split:] if split < n else perm[-1:]

    in_dim = X.shape[1]
    W1 = (rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    W2_state = (rng.standard_normal((hidden_dim, EMBED_DIM)) * np.sqrt(1.0 / hidden_dim)).astype(np.float32)
    b2_state = np.zeros((EMBED_DIM,), dtype=np.float32)
    W2_risk = (rng.standard_normal((hidden_dim, 1)) * np.sqrt(1.0 / hidden_dim)).astype(np.float32)
    b2_risk = np.zeros((1,), dtype=np.float32)

    def evaluate(idx: np.ndarray) -> Dict[str, float]:
        h = np.maximum(0.0, X[idx] @ W1 + b1)
        z_raw = h @ W2_state + b2_state
        z_norm = z_raw / np.maximum(np.linalg.norm(z_raw, axis=1, keepdims=True), 1e-8)
        cos_sim = np.sum(z_norm * Y_state[idx], axis=1)
        state_surprise = float(np.mean(1.0 - cos_sim))
        p_risk = sigmoid(h @ W2_risk + b2_risk).reshape(-1)
        risk_mae = float(np.mean(np.abs(p_risk - Y_risk[idx])))
        return {"state_surprise": round(state_surprise, 4), "risk_mae": round(risk_mae, 4)}

    val_before = evaluate(va_idx)
    t0 = time.perf_counter()

    X_tr, Ys_tr, Yr_tr = X[tr_idx], Y_state[tr_idx], Y_risk[tr_idx].reshape(-1, 1)
    m = float(len(tr_idx))

    for _ in range(epochs):
        # Forward
        z1 = X_tr @ W1 + b1
        h = np.maximum(0.0, z1)
        z_pred = h @ W2_state + b2_state
        r_logit = h @ W2_risk + b2_risk
        r_pred = sigmoid(r_logit)

        # Gradients: MSE in normalized latent space + BCE on stall/drift risk
        dz_pred = (2.0 / m) * (z_pred - Ys_tr)
        dr_logit = (1.0 / m) * (r_pred - Yr_tr)

        dW2_state = h.T @ dz_pred
        db2_state = np.sum(dz_pred, axis=0)
        dW2_risk = h.T @ dr_logit
        db2_risk = np.sum(dr_logit, axis=0)

        dh = dz_pred @ W2_state.T + dr_logit @ W2_risk.T
        dz1 = dh * (z1 > 0.0).astype(np.float32)
        dW1 = X_tr.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # SGD with mild L2 weight decay
        wd = 1e-4
        W1 -= lr * (dW1 + wd * W1)
        b1 -= lr * db1
        W2_state -= lr * (dW2_state + wd * W2_state)
        b2_state -= lr * db2_state
        W2_risk -= lr * (dW2_risk + wd * W2_risk)
        b2_risk -= lr * db2_risk

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    val_after = evaluate(va_idx)
    train_after = evaluate(tr_idx)

    # Rank validation examples by remaining surprise (top candidates for new rules!)
    h_va = np.maximum(0.0, X[va_idx] @ W1 + b1)
    z_va = h_va @ W2_state + b2_state
    z_va = z_va / np.maximum(np.linalg.norm(z_va, axis=1, keepdims=True), 1e-8)
    va_surprises = 1.0 - np.sum(z_va * Y_state[va_idx], axis=1)

    weights = {
        "W1": W1,
        "b1": b1,
        "W2_state": W2_state,
        "b2_state": b2_state,
        "W2_risk": W2_risk,
        "b2_risk": b2_risk,
    }
    metrics = {
        "episodes_total": int(n),
        "train_count": int(len(tr_idx)),
        "val_count": int(len(va_idx)),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "train_ms": elapsed_ms,
        "val_before": val_before,
        "val_after": val_after,
        "train_after": train_after,
        "max_val_surprise": round(float(np.max(va_surprises)), 4),
    }
    return weights, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny local 384-D JEPA predictor from residual episodes.")
    parser.add_argument("--episodes", type=Path, default=Path("episodes/transitions.jsonl"), help="Input episodes NDJSON")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/my_m2_jepa.npz"), help="Output checkpoint .npz")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--demo-synthetic", action="store_true", help="Train on synthetic Day-1 demo episodes")
    parser.add_argument("--force-hash", action="store_true", help="Use zero-download hash384 embedder")
    args = parser.parse_args()

    embedder = LocalEmbedder(force_hash=args.force_hash)
    if args.demo_synthetic:
        episodes = build_demo_synthetic_episodes(40)
        source_label = "synthetic_demo (40 episodes)"
    else:
        if not args.episodes.exists():
            print(f"Episodes file not found: {args.episodes}. Run 02_watch_frontier_agents.py or pass --demo-synthetic.")
            return 1
        episodes = load_episodes(args.episodes)
        source_label = f"{args.episodes} ({len(episodes)} episodes)"

    if len(episodes) < 6:
        print(f"Need at least 6 episodes to train/validate cleanly (found {len(episodes)}). Pass --demo-synthetic to test now.")
        return 1

    curr_texts = [str(e["current_state"].get("summary", "")) for e in episodes]
    intent_texts = [str(e["intended_outcome"].get("summary", "")) for e in episodes]
    end_texts = [str(e["end_state"].get("summary", "")) for e in episodes]
    risk_targets = np.array(
        [float(e.get("assessment_vs_expectations", {}).get("actual_risk_target", e.get("end_state", {}).get("thrash_score", 0.0))) for e in episodes],
        dtype=np.float32,
    )

    z_curr = embedder.encode(curr_texts)
    z_intent = embedder.encode(intent_texts)
    z_end = embedder.encode(end_texts)
    X = np.concatenate([z_curr, z_intent], axis=1)

    weights, metrics = train_tiny_jepa(X, z_end, risk_targets, epochs=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **weights)

    print(f"=== Tiny Local JEPA Training Summary ({source_label}) ===")
    print(f"Embedder       : {embedder.backend_name}")
    print(f"Train / Val    : {metrics['train_count']} / {metrics['val_count']} episodes ({metrics['epochs']} epochs in {metrics['train_ms']} ms)")
    print(f"Val BEFORE     : state_surprise={metrics['val_before']['state_surprise']:.4f} | risk_mae={metrics['val_before']['risk_mae']:.4f}")
    print(f"Val AFTER      : state_surprise={metrics['val_after']['state_surprise']:.4f} | risk_mae={metrics['val_after']['risk_mae']:.4f}")
    print(f"Saved weights  -> {args.out}")
    print(f"Next: pass `--jepa-checkpoint {args.out}` to `02_watch_frontier_agents.py` for live predictions!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
