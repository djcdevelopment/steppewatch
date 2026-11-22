#!/usr/bin/env python3
"""Shared local 384-D embedder, hardware probe, and semantic thrash detector.

Works immediately with only `numpy` (via a deterministic 384-D word/char n-gram
hash encoder for fast zero-download math checks), and automatically upgrades to
real semantic embeddings (`all-MiniLM-L6-v2` via `sentence-transformers` on Apple
Silicon MPS or CPU) when installed.
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

EMBED_DIM = 384
DEFAULT_THRASH_THRESHOLD = 0.65

STEP_PREFIX_RE = re.compile(
    r"^\s*(?:step|turn|attempt|iter(?:ation)?)\s*\d+\s*[:.\-]\s*", re.I
)
ERROR_SIGNATURE_RE = re.compile(
    r"\b(error|exception|traceback|failed|refused|timeout|timed out|"
    r"syntaxerror|keyerror|valueerror|typeerror|assertionerror|runtimeerror|"
    r"connectionrefusederror|rc=[1-9]|http\s+[45]\d\d)\b",
    re.I,
)
SPECIFIC_EXCEPTION_RE = re.compile(
    r"\b([A-Za-z]+Error|HTTP\s+\d{3}|CS\d{4}|TS\d{4}|F\d{3}|rc=[1-9]\d*|line\s+\d+)\b"
)
CONCRETE_FIX_ACTION_RE = re.compile(
    r"\b(applied\s+fix|fixed|fix\s+the|updated|edited|added\s+[A-Za-z_]|"
    r"patched|implemented|created|wrote|resolved|refactored|ran\s+(?:pytest|dotnet|npm|cargo))\b",
    re.I,
)
RESOLUTION_SUCCESS_RE = re.compile(
    r"\b(passed|0\s+errors|exit\s+code\s+0|http\s+200|test\s+result:\s*ok|"
    r"accepted|verified|succeeded|all\s+\d+\s+tests\s+passed)\b",
    re.I,
)
ENGLISH_SENTENCE_GERUNDS: Set[str] = {
    "inspecting", "looking", "reviewing", "examining", "checking", "reading",
    "taking", "repeating", "running", "retrying", "regenerating", "generating",
    "attempting", "trying", "verifying", "confirming", "searching", "scanning",
    "tracing", "analyzing", "evaluating", "revisiting", "reconsidering",
    "let", "please", "still", "again", "another", "every", "each", "step",
    "pass", "turn",
}


def probe_hardware() -> Dict[str, Any]:
    """Detect local OS, CPU, Apple Silicon details, and installed ML backends."""
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "backends": {"numpy": True, "mlx": False, "torch_mps": False, "onnx_coreml": False},
    }
    if platform.system() == "Darwin":
        for key, sysctl_name in (
            ("mac_model", "hw.model"),
            ("chip", "machdep.cpu.brand_string"),
            ("mem_bytes", "hw.memsize"),
        ):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", sysctl_name], text=True, stderr=subprocess.DEVNULL
                ).strip()
                info[key] = int(out) if key == "mem_bytes" else out
            except Exception:
                pass
        if "mem_bytes" in info:
            info["mem_gib"] = round(info["mem_bytes"] / (1024 ** 3), 1)

    try:
        import mlx.core  # type: ignore  # noqa: F401
        info["backends"]["mlx"] = True
    except Exception:
        pass

    try:
        import torch  # type: ignore
        info["backends"]["torch_mps"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    try:
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        info["backends"]["onnx_coreml"] = "CoreMLExecutionProvider" in providers
        info["onnx_providers"] = providers
    except Exception:
        pass

    return info


class LocalEmbedder:
    """384-D text embedder with automatic SentenceTransformer -> hash fallback."""

    def __init__(self, force_hash: bool = False, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dim = EMBED_DIM
        self.backend_name = "hash384_cpu"
        self._st_model: Any = None
        if not force_hash:
            try:
                import torch  # type: ignore
                from sentence_transformers import SentenceTransformer  # type: ignore

                device = (
                    "mps"
                    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                    else "cpu"
                )
                self._st_model = SentenceTransformer(model_name, device=device)
                self.backend_name = f"sentence_transformers:{model_name}:{device}"
            except Exception:
                self._st_model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape [len(texts), 384]."""
        items = [str(t or "")[:1200] for t in texts]
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._st_model is not None:
            embs = self._st_model.encode(
                items,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(embs, dtype=np.float32)

        return self._encode_hash384(items)

    def _encode_hash384(self, items: Sequence[str]) -> np.ndarray:
        """Deterministic 384-D feature-hashed word + char-trigram embeddings.

        Zero downloads and zero ML dependencies — useful for verifying matrix
        throughput and lexical/paraphrase geometry before downloading MiniLM.
        """
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(items):
            vec = out[row_idx]
            low = text.lower()
            words = re.findall(r"[a-z0-9_./:-]{2,}", low)
            features: List[Tuple[str, float]] = []
            for w in words:
                features.append((f"w:{w}", 1.5))
                if len(w) >= 4:
                    for i in range(len(w) - 2):
                        features.append((f"c:{w[i:i+3]}", 0.45))
            for i in range(len(words) - 1):
                features.append((f"b:{words[i]}_{words[i+1]}", 1.1))
            for feat, weight in features:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                val = int.from_bytes(digest, "little")
                bucket = val % self.dim
                sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
                vec[bucket] += sign * weight
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
        return out


def extract_true_code_targets(step_text: str) -> Set[str]:
    """Extract concrete code targets while ignoring English sentence-starter prose."""
    cleaned = STEP_PREFIX_RE.sub("", step_text)
    targets: Set[str] = set()

    # 1. File names (code, config, docs)
    for fn in re.findall(
        r"\b([A-Za-z0-9_.\-]+\.(?:py|cs|ts|tsx|js|jsx|go|rs|java|cpp|h|json|yaml|yml|toml|md|sh))\b",
        cleaned,
    ):
        targets.add(fn.lower())

    # 2. Explicit function/method calls foo()
    for fn_call in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", cleaned):
        if fn_call.lower() not in ENGLISH_SENTENCE_GERUNDS:
            targets.add(fn_call.lower())

    # 3. snake_case identifiers or UPPER_CASE constants
    for sc in re.findall(
        r"\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b|\b[A-Z][A-Z0-9_]{2,}\b", cleaned
    ):
        targets.add(sc.lower())

    # 4. camelCase / PascalCase code symbols
    for cc in re.findall(r"\b[A-Za-z0-9]+[a-z0-9][A-Z][A-Za-z0-9]*\b", cleaned):
        if cc.lower() not in ENGLISH_SENTENCE_GERUNDS:
            targets.add(cc.lower())

    # 5. Line numbers, compiler diagnostics, HTTP codes, exit codes
    for loc in re.findall(
        r"\b(?:line\s+\d+|http\s+\d{3}|cs\d{4}|ts\d{4}|f\d{3}|rc=\d+)\b",
        cleaned,
        flags=re.I,
    ):
        targets.add(re.sub(r"\s+", "_", loc.lower()))

    return targets


def detect_agent_thrash(
    embedder: LocalEmbedder,
    steps: Sequence[str],
    trace_id: str = "trace-0",
    thrash_threshold: float = DEFAULT_THRASH_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate the last 3-6 agent trace steps for semantic loops or polite stalls."""
    t0 = time.perf_counter()
    raw_window = [s.strip()[:600] for s in steps[-6:] if s and s.strip()]
    if len(raw_window) < 3:
        return {
            "trace_id": trace_id,
            "thrash_detected": False,
            "thrash_score": 0.0,
            "thrash_threshold": thrash_threshold,
            "action": "continue",
            "reason": "fewer_than_3_steps",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "backend": embedder.backend_name,
        }

    norm_window = [STEP_PREFIX_RE.sub("", s).strip() or s for s in raw_window]
    embs = embedder.encode(norm_window)
    sim_mat = embs @ embs.T

    adj_cos = [float(sim_mat[i, i + 1]) for i in range(len(norm_window) - 1)]
    osc_cos = [float(sim_mat[i, i + 2]) for i in range(len(norm_window) - 2)]
    upper_pairs = [
        float(sim_mat[i, j])
        for i in range(len(norm_window))
        for j in range(i + 1, len(norm_window))
    ]

    mean_adj_cos = float(np.mean(adj_cos))
    max_osc_cos = float(np.max(osc_cos)) if osc_cos else 0.0
    all_pairs_cos = float(np.mean(upper_pairs)) if upper_pairs else mean_adj_cos
    effective_loop_cos = max(mean_adj_cos, 0.94 * max_osc_cos, all_pairs_cos)

    err_hits = [1 if ERROR_SIGNATURE_RE.search(s) else 0 for s in norm_window]
    raw_error_ratio = sum(err_hits) / float(len(norm_window))
    step_err_sigs = [
        frozenset(m.lower() for m in SPECIFIC_EXCEPTION_RE.findall(s)) for s in norm_window
    ]
    distinct_err_sigs = len({sig for sig in step_err_sigs if sig})

    step_target_sets = [extract_true_code_targets(s) for s in norm_window]
    seen_targets: Set[str] = set()
    new_target_counts: List[int] = []
    for t_set in step_target_sets:
        new_t = t_set - seen_targets
        new_target_counts.append(len(new_t))
        seen_targets.update(t_set)

    tail_new_targets = sum(new_target_counts[1:])
    zero_new_target_tail_steps = sum(1 for c in new_target_counts[1:] if c == 0)
    stagnation_ratio = zero_new_target_tail_steps / float(max(1, len(norm_window) - 1))

    fix_steps = sum(1 for s in norm_window[1:] if CONCRETE_FIX_ACTION_RE.search(s))
    final_step_resolved = bool(RESOLUTION_SUCCESS_RE.search(norm_window[-1])) and not bool(
        re.search(r"\b(failed|error=[1-9]|rc=[1-9]|unresolved)\b", norm_window[-1], re.I)
    )

    productive_debug = (
        (distinct_err_sigs >= 3 and fix_steps >= 2)
        or (final_step_resolved and fix_steps >= 1 and tail_new_targets >= 2)
    )

    if productive_debug:
        effective_error_ratio = 0.0
        progress_score = min(1.0, (tail_new_targets + 2 * fix_steps) / 6.0)
    else:
        effective_error_ratio = raw_error_ratio
        progress_score = min(1.0, tail_new_targets / max(3.0, float(len(norm_window) * 1.5)))

    # Hash fallback has slightly lower baseline cosine than MiniLM on paraphrases;
    # keep MiniLM thresholds canonical while adapting gracefully if hash fallback is used.
    cos_floor = 0.36 if embedder.backend_name == "hash384_cpu" else 0.56
    cos_span = 0.28
    silent_floor = 0.38 if embedder.backend_name == "hash384_cpu" else 0.58

    cos_component = max(0.0, min(1.0, (effective_loop_cos - cos_floor) / cos_span))
    silent_loop_boost = 0.0
    if (
        effective_loop_cos >= silent_floor
        and stagnation_ratio >= 0.66
        and fix_steps == 0
        and not final_step_resolved
    ):
        silent_loop_boost = 0.44 * stagnation_ratio

    thrash_score = float(
        np.clip(
            0.62 * cos_component
            + 0.34 * effective_error_ratio
            + silent_loop_boost
            + (0.14 if effective_loop_cos >= 0.84 else 0.0)
            - 0.36 * progress_score
            - (0.35 if productive_debug else 0.0),
            0.0,
            1.0,
        )
    )
    thrash_detected = thrash_score >= thrash_threshold
    return {
        "trace_id": trace_id,
        "thrash_detected": thrash_detected,
        "thrash_score": round(thrash_score, 4),
        "thrash_threshold": thrash_threshold,
        "action": "early_abort_and_reroute" if thrash_detected else "continue",
        "mean_adj_cosine": round(mean_adj_cos, 4),
        "max_osc_cosine": round(max_osc_cos, 4),
        "all_pairs_cosine": round(all_pairs_cos, 4),
        "error_repeat_ratio": round(raw_error_ratio, 4),
        "stagnation_ratio": round(stagnation_ratio, 4),
        "tail_new_targets": tail_new_targets,
        "productive_debug_progression": productive_debug,
        "progress_score": round(progress_score, 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "backend": embedder.backend_name,
    }


def predict_with_jepa_checkpoint(
    ckpt_path: str, current_emb: np.ndarray, intent_emb: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Run forward pass of a saved NumPy `.npz` tiny JEPA predictor."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    data = np.load(ckpt_path)
    x = np.concatenate([current_emb.astype(np.float32), intent_emb.astype(np.float32)], axis=-1)
    h = np.maximum(0.0, x @ data["W1"] + data["b1"])
    z_pred = h @ data["W2_state"] + data["b2_state"]
    norm = float(np.linalg.norm(z_pred))
    if norm > 1e-8:
        z_pred = z_pred / norm
    risk_logit = float((h @ data["W2_risk"] + data["b2_risk"]).reshape(-1)[0])
    risk_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, risk_logit))))
    return {
        "predicted_end_emb": z_pred.astype(np.float32),
        "predicted_stall_risk": round(risk_prob, 4),
        "checkpoint": ckpt_path,
    }
