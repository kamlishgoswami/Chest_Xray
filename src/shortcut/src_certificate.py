"""Shortcut Reliance Certificate (SRC) — the auditable artifact, C2 (PAPER_OUTLINE.md §3.4).

Distills the per-channel causal effects from the CSA (csa.py) into a single per-model
certificate with a per-channel breakdown, confidence interval, and a machine-checkable
VALIDITY gate built from the control channels.

SRC definition (NORMALIZED — the headline metric):
    SRC = mean(shortcut-channel effects) / inside_lung effect.

    RATIONALE (proven empirically on the 12-epoch run): the raw mean-of-effects is confounded with
    model accuracy (corr ~0.90) — a more confident model loses more confidence on EVERY intervention,
    so the raw mean just re-measures accuracy and adds nothing over a trivial baseline (partial-r|acc
    ~0.02). DIVIDING by the inside-lung (real-pathology) effect cancels that confound: it asks "does
    the model lean on shortcuts MORE than on real anatomy?" — accuracy-independent by construction.
    This normalized SRC has strong independent signal (partial-r|acc ~0.5-0.6), survives dropping the
    weak models, and predicts post-temperature-scaling miscalibration better than the raw mean.
    We report `src_raw_mean` alongside for an honest before/after comparison.

Validity gate (§3.3.3) — the soundness mechanism, made checkable:
    valid = (|sham effect| <= SHAM_TOL)              # negative control must be ~null
            and (inside_lung effect >= LUNG_MIN)     # positive control must bite (also the denominator)
    An invalid certificate means the audit cannot be trusted for that model/run. The gate ALSO
    guarantees the normalization denominator is non-trivial (inside_lung >= LUNG_MIN).

The SRC is the predictor in the C3 coupling result (predicts cross-source collapse + ECE).
"""
from __future__ import annotations

import json
from pathlib import Path

from .csa import SHORTCUT_CHANNELS, CONTROL_CHANNELS

SHAM_TOL = 0.02   # |sham effect| must be <= this  (negative control)
LUNG_MIN = 0.05   # inside_lung effect should be >= this (positive control AND normalization denom)


def compute_src(audit_result):
    """Aggregate CSA channel effects into the per-model SRC certificate dict.

    `audit_result` is the {channel: {effect, ci_lo, ci_hi, n}} mapping from csa.run_audit.
    Returns a dict with the NORMALIZED SRC (headline), the raw mean (for comparison), per-channel
    breakdown, CI, and the validity gate.
    """
    shortcut = {c: audit_result[c] for c in SHORTCUT_CHANNELS if c in audit_result}
    if not shortcut:
        raise ValueError("audit_result has no shortcut channels; run csa.run_audit first")

    effects = [max(0.0, r["effect"]) for r in shortcut.values()]  # reliance is non-negative
    raw_mean = sum(effects) / len(effects)

    sham = audit_result.get("sham", {}).get("effect", float("nan"))
    inside = audit_result.get("inside_lung", {}).get("effect", float("nan"))
    valid = (abs(sham) <= SHAM_TOL) and (inside >= LUNG_MIN)

    # NORMALIZED SRC = shortcut reliance RELATIVE to real-pathology reliance (accuracy-independent).
    # Guard the denominator: if inside_lung is tiny/non-positive the ratio is undefined -> the
    # validity gate (inside >= LUNG_MIN) is what makes this trustworthy; for invalid models we still
    # emit a value but it should be filtered out downstream via `valid`.
    denom = max(inside, LUNG_MIN) if inside == inside else float("nan")  # NaN-safe
    src_norm = float(raw_mean / denom) if denom and denom == denom else float("nan")

    # CI on the normalized SRC: propagate the shortcut-mean CI through the same denominator
    ci_lo_raw = sum(max(0.0, r["ci_lo"]) for r in shortcut.values()) / len(shortcut)
    ci_hi_raw = sum(max(0.0, r["ci_hi"]) for r in shortcut.values()) / len(shortcut)
    ci_lo = float(ci_lo_raw / denom) if denom and denom == denom else float("nan")
    ci_hi = float(ci_hi_raw / denom) if denom and denom == denom else float("nan")

    return {
        "src": float(src_norm),                 # HEADLINE: normalized (shortcut / inside_lung)
        "src_raw_mean": float(raw_mean),         # old metric, kept for honest comparison/reporting
        "src_ci_lo": ci_lo,
        "src_ci_hi": ci_hi,
        "normalization": "mean(shortcut_effects) / inside_lung_effect",
        "per_channel": {c: float(r["effect"]) for c, r in shortcut.items()},
        "per_channel_ci": {c: [float(r["ci_lo"]), float(r["ci_hi"])] for c, r in shortcut.items()},
        "controls": {c: float(audit_result[c]["effect"]) for c in CONTROL_CHANNELS if c in audit_result},
        "valid": bool(valid),
        "validity_detail": {
            "sham_effect": float(sham), "sham_tol": SHAM_TOL,
            "inside_lung_effect": float(inside), "inside_lung_min": LUNG_MIN,
        },
    }


def emit_certificate(model_name, audit_result, path):
    """Compute SRC and write the auditable certificate JSON to `path`. Returns the cert dict."""
    cert = compute_src(audit_result)
    cert["model"] = model_name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cert, indent=2))
    return cert


def dominant_channel(cert):
    """Name of the shortcut channel the model relies on most (for reporting/figures)."""
    pc = cert["per_channel"]
    return max(pc, key=pc.get) if pc else None
