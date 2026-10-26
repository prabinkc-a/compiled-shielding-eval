"""Supervision-separated finite-program policy optimization, using NumPy.

Training receives a TrainSignals object with NO true accuracy or exploit label.
All methods use the same sampled REINFORCE update and initial logits.
"""
from dataclasses import dataclass
import time
import numpy as np

MAIN = ["baseline", "A_rta", "B_nyaya", "C_process", "D_hierarchy", "E_unified"]
ABLATIONS = ["A_full_constraints", "C_progress", "E_no_gate", "E_no_process",
             "E_no_hierarchy", "E_gate_only", "baseline_penalty_1",
             "baseline_penalty_5", "oracle_outcome"]


@dataclass
class TrainSignals:
    proxy: np.ndarray
    hard: np.ndarray
    causal: np.ndarray
    process: np.ndarray
    progress: np.ndarray
    iso: np.ndarray
    cost: np.ndarray
    work: np.ndarray
    verify_work: np.ndarray


def softmax(logits, allowed):
    z = np.where(allowed, logits, -np.inf)
    p = np.exp(z - z.max())
    return p / p.sum()


def tangent_project(gradient, allowed):
    """Orthogonal projection on {v: v_forbidden=0, sum(v_allowed)=0}."""
    out = np.zeros_like(gradient)
    out[allowed] = gradient[allowed] - gradient[allowed].mean()
    return out


def supervision(signals, method, gate_miss=0.0, gate_false_reject=0.0, rng=None):
    s = signals
    allowed = np.ones(s.proxy.shape[1], dtype=bool)
    direct = np.max(s.hard, axis=0) == 0
    global_ok = np.max(np.maximum(s.hard, s.causal), axis=0) == 0
    if method == "A_rta":
        allowed &= direct
    elif method in ("D_hierarchy", "E_unified", "E_no_gate", "E_no_process", "A_full_constraints"):
        # Exact lexicographic minimum violation is zero, achieved by abstention.
        # Finite, known program contracts permit exhaustive support filtering.
        allowed &= global_ok
    elif method == "E_no_hierarchy":
        allowed &= direct
    if not allowed.any():
        raise RuntimeError("No certified feasible program: refuse rather than relax integrity.")
    gate = (1 - s.hard) * (1 - s.causal) * s.iso
    if gate_miss or gate_false_reject:
        if rng is None:
            raise ValueError("Noise requires a random generator")
        u = rng.random(gate.shape)
        gate = np.where(gate == 0, (u < gate_miss).astype(float),
                        (u >= gate_false_reject).astype(float))
    uses_gate = method in ("B_nyaya", "E_unified", "E_no_process", "E_no_hierarchy", "E_gate_only")
    uses_process = method in ("E_unified", "E_no_gate", "E_no_hierarchy")
    if method == "C_process":
        rewards = s.process.copy()
    elif method == "C_progress":
        rewards = s.process + 0.3 * s.progress
    elif uses_gate:
        rewards = s.proxy * gate - s.cost
    else:
        penalty = 1.0 if method == "baseline_penalty_1" else 5.0 if method == "baseline_penalty_5" else 0.1
        rewards = s.proxy - s.cost - penalty * s.hard
    if uses_process:
        rewards += 0.15 * s.progress * (1 - s.hard) * (1 - s.causal)
    return rewards, allowed, uses_gate


def train(signals, method, seed, updates=600, batch_size=64, lr=2.0,
          entropy=0.004, gate_miss=0.0, gate_false_reject=0.0):
    started = time.perf_counter()
    # Separate noise and rollout streams keep rollout draws paired across methods.
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 9271)
    rewards, allowed, uses_gate = supervision(signals, method, gate_miss,
                                               gate_false_reject, noise_rng)
    logits = rng.normal(0, 0.01, rewards.shape[1])
    n, k = rewards.shape
    training_work, verification_work, learning_curve = 0.0, 0.0, []
    # Bounds/certificate checking is charged once for this finite library.
    certificate_work = float(n * k * (2 if method in (
        "D_hierarchy", "E_unified", "E_no_gate", "E_no_process", "A_full_constraints")
        else 1)) if not allowed.all() else 0.0
    for step in range(updates):
        p = softmax(logits, allowed)
        tasks = rng.integers(n, size=batch_size)
        actions = np.searchsorted(np.cumsum(p), rng.random(batch_size), side="right")
        actions = np.minimum(actions, k - 1)
        r = rewards[tasks, actions]
        # Leave-one-out baseline; unbiased group-centred REINFORCE estimator.
        advantage = (r - r.mean()) * batch_size / (batch_size - 1)
        g = np.bincount(actions, weights=advantage, minlength=k) / batch_size
        g -= p * advantage.mean()
        logp = np.log(np.maximum(p, 1e-300))
        g += entropy * (-p * (logp - np.dot(p, logp)))
        if method.startswith("A_"):
            g = tangent_project(g, allowed)
        else:
            g[~allowed] = 0
        logits += lr * g
        logits -= logits[allowed].mean()
        training_work += float(signals.work[tasks, actions].sum())
        if uses_gate:
            verification_work += float(signals.verify_work[tasks, actions].sum())
        if step % 50 == 0 or step == updates - 1:
            learning_curve.append([step + 1, float(p @ rewards.mean(axis=0)),
                                   float(p @ signals.proxy.mean(axis=0))])
    return {
        "p": softmax(logits, allowed), "seconds": time.perf_counter() - started,
        "base_work": training_work, "verify_work": verification_work,
        "certificate_work": certificate_work,
        "allowed_count": int(allowed.sum()), "curve": learning_curve,
    }
