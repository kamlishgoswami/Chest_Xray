"""Shortcut Reliance Certificate (SRC) — the auditable artifact, C2 (PAPER_OUTLINE.md §3.4).

Distills the per-channel causal effects from the CSA (csa.py) into a single per-model
certificate with a per-channel breakdown, confidence interval, and a machine-checkable
VALIDITY gate built from the control channels.

SRC definition:
    SRC = mean of the shortcut-channel effects (border, background, source_signature),
          clipped to [0, 1]. Higher = the model relies more on shortcuts.
    Per-channel contributions are reported so reliance can be attributed.

Validity gate (§3.3.3) — the soundness mechanism, made checkable:
    valid = (|sham effect| <= SHAM_TOL)              # negative control must be ~null
            and (inside_lung effect >= LUNG_MIN)     # positive control must bite
    An invalid certificate means the audit cannot be trusted for that model/run.

The SRC is the predictor in the C3 coupling result (predicts cross-source collapse + ECE).
"""
from __future__ import annotations

import json
from pathlib import Path

from .csa import SHORTCUT_CHANNELS, CONTROL_CHANNELS

SHAM_TOL = 0.02   # |sham effect| must be <= this  (negative control)
LUNG_MIN = 0.05   # inside_lung effect should be >= this (positive control sanity)


def compute_src(audit_result):
    """Aggregate CSA channel effects into the per-model SRC certificate dict.

    `audit_result` is the {channel: {effect, ci_lo, ci_hi, n}} mapping from csa.run_audit.
    Returns a dict with overall SRC, per-channel breakdown, CI, and the validity gate.
    """
    shortcut = {c: audit_result[c] for c in SHORTCUT_CHANNELS if c in audit_result}
    if not shortcut:
        raise ValueError("audit_result has no shortcut channels; run csa.run_audit first")

    effects = [max(0.0, r["effect"]) for r in shortcut.values()]  # reliance is non-negative
    src = min(1.0, sum(effects) / len(effects))

    # propagate uncertainty: average the per-channel CI bounds (clipped to [0,1])
    ci_lo = min(1.0, max(0.0, sum(max(0.0, r["ci_lo"]) for r in shortcut.values()) / len(shortcut)))
    ci_hi = min(1.0, max(0.0, sum(max(0.0, r["ci_hi"]) for r in shortcut.values()) / len(shortcut)))

    sham = audit_result.get("sham", {}).get("effect", float("nan"))
    inside = audit_result.get("inside_lung", {}).get("effect", float("nan"))
    valid = (abs(sham) <= SHAM_TOL) and (inside >= LUNG_MIN)

    return {
        "src": float(src),
        "src_ci_lo": float(ci_lo),
        "src_ci_hi": float(ci_hi),
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
