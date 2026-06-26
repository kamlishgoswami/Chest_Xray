"""Cross-source evaluation — the OUTCOME that SRC predicts (PAPER_OUTLINE.md §4.6, §6).

For each model we measure performance on the in-domain test split and on the
held-out CROSS-SOURCE test set (same diseases, different acquisition source), and
report the accuracy collapse (delta_acc) and the change in calibration (delta_ece).
These deltas are the dependent variables in the C3 predictive-coupling result
(SRC -> collapse + miscalibration).

`confounder_separation()` (§4.7) decomposes the collapse into a shortcut-attributable
component (explained by SRC) vs. a genuine pathology-distribution-shift residual.

TERMINOLOGY (§4.8): this module handles (a) cross-source/domain shift — the real
source change. It is NOT (b) covariate-shift perturbations (src/robustness, SSP,
corroborating only) nor (c) counterfactual interventions (csa.py, the method).

HONEST DATA NOTE (verified against data/manifest.csv, 2026-06): cross-source coverage
is uneven — Covid (COVID-QU-Ex), Normal (Montgomery) and TB (Montgomery) have a
cross_source split; Pneumonia does NOT until RSNA is downloaded (it is marked optional
in configs/datasets.yaml). So the cross-source test set is the subset of classes that
actually have cross_source rows. This is reported, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _metrics_on(model, df, batch_size=32):
    """Run a model over the rows in `df`; return (accuracy, ece, n, classes_present)."""
    from src.data.loaders import make_dataset, CLASS_TO_IDX
    from src.evaluation.metrics import standard_metrics, calibration_metrics

    if len(df) == 0:
        return {"accuracy": float("nan"), "ece": float("nan"), "n": 0, "classes": []}

    ds = make_dataset(df, batch_size=batch_size, training=False, shuffle=False)
    y_true = np.array([CLASS_TO_IDX[c] for c in df["disease"]])
    y_prob = model.predict(ds, verbose=0)

    m = {**standard_metrics(y_true, y_prob), **calibration_metrics(y_true, y_prob)}
    return {
        "accuracy": float(m["accuracy"]),
        "ece": float(m["ece"]),
        "n": int(len(df)),
        "classes": sorted(df["disease"].unique().tolist()),
    }


def run_cross_source_matrix(models=None, batch_size=32, results_dir=None):
    """In-domain vs cross-source accuracy/ECE per model -> collapse deltas.

    For every checkpointed model: evaluate on the in-domain test split AND on the
    cross-source rows (restricted to classes that have cross_source coverage, so the
    two sets are comparable on the same class subset). Writes
    results/cross_source.json and returns the list of per-model dicts.

    delta_acc = in_domain_acc - cross_source_acc   (positive = collapse)
    delta_ece = cross_source_ece - in_domain_ece   (positive = worse calibration)
    """
    import tensorflow as tf
    from src.data.loaders import load_manifest, filter_rows

    results_dir = Path(results_dir) if results_dir else (ROOT / "results")

    df = load_manifest()
    cross_df = filter_rows(df, roles=["cross_source"])
    cross_classes = sorted(cross_df["disease"].unique().tolist())
    # restrict in-domain test to the SAME classes so collapse is not confounded by
    # which classes happen to have a cross-source set.
    indom_df = filter_rows(df, split="test", roles=["in_domain"])
    indom_df = indom_df[indom_df["disease"].isin(cross_classes)]

    model_dirs = ([results_dir / m for m in models] if models
                  else [d for d in results_dir.iterdir() if d.is_dir()])

    rows = []
    for mdir in model_dirs:
        name = mdir.name
        ckpt = mdir / f"{name}_best.keras"
        if not ckpt.exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        print(f"=== cross-source: {name} ===")
        model = tf.keras.models.load_model(ckpt)

        indom = _metrics_on(model, indom_df, batch_size)
        cross = _metrics_on(model, cross_df, batch_size)

        rec = {
            "model": name,
            "cross_source_classes": cross_classes,
            "in_domain_acc": indom["accuracy"], "in_domain_ece": indom["ece"], "in_domain_n": indom["n"],
            "cross_source_acc": cross["accuracy"], "cross_source_ece": cross["ece"], "cross_source_n": cross["n"],
            "delta_acc": float(indom["accuracy"] - cross["accuracy"]),
            "delta_ece": float(cross["ece"] - indom["ece"]),
        }
        rows.append(rec)
        print(f"  in-domain acc={indom['accuracy']:.3f} ece={indom['ece']:.3f} | "
              f"cross acc={cross['accuracy']:.3f} ece={cross['ece']:.3f} | "
              f"Δacc={rec['delta_acc']:+.3f} Δece={rec['delta_ece']:+.3f}")

    out = results_dir / "cross_source.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\ncross-source -> {out.relative_to(ROOT)}")
    return rows


def _load_src_scores(models, results_dir):
    """Read each model's SRC from results/<model>/certificate.json. Returns {model: src}."""
    src = {}
    for mdir in models:
        cert = Path(mdir) / "certificate.json"
        if cert.exists():
            src[Path(mdir).name] = json.loads(cert.read_text()).get("src")
    return src


def couple_src_to_collapse(results_dir=None):
    """C3 — regress cross-source collapse (delta_acc, delta_ece) on SRC across models.

    Reads results/cross_source.json + each results/<model>/certificate.json, pairs
    (SRC, delta_acc) and (SRC, delta_ece), and reports Pearson r, slope, intercept,
    and R^2 for each. With <3 models it reports the raw pairs only (a regression on
    2 points is meaningless) — this is the §8b preliminary go/no-go check.

    Returns a dict; also writes results/c3_coupling.json.
    """
    results_dir = Path(results_dir) if results_dir else (ROOT / "results")
    cs_path = results_dir / "cross_source.json"
    if not cs_path.exists():
        raise FileNotFoundError("run run_cross_source_matrix() first (results/cross_source.json missing)")

    cs = json.loads(cs_path.read_text())
    cert_src = _load_src_scores([results_dir / r["model"] for r in cs], results_dir)

    pairs = [(cert_src.get(r["model"]), r["delta_acc"], r["delta_ece"], r["model"])
             for r in cs if cert_src.get(r["model"]) is not None]

    out = {"n_models": len(pairs),
           "pairs": [{"model": m, "src": s, "delta_acc": da, "delta_ece": de}
                     for (s, da, de, m) in pairs]}

    if len(pairs) >= 3:
        src = np.array([p[0] for p in pairs], float)
        for dep_name, idx in (("delta_acc", 1), ("delta_ece", 2)):
            y = np.array([p[idx] for p in pairs], float)
            out[dep_name] = _ols_fit(src, y)
    else:
        out["note"] = ("Fewer than 3 models with both SRC and cross-source results; "
                       "showing raw pairs only. Train >=3 models for the §8b coupling check.")

    (results_dir / "c3_coupling.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _ols_fit(x, y):
    """Simple OLS y ~ x with Pearson r, slope, intercept, R^2. Pure numpy (no deps)."""
    if np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return {"r": float("nan"), "slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "note": "zero variance in x or y"}
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    return {"r": r, "slope": float(slope), "intercept": float(intercept), "r2": float(r * r)}


def confounder_separation(results_dir=None):
    """§4.7 — split cross-source collapse into shortcut-attributable vs genuine-shift residual.

    Uses the C3 regression: the variance in delta_acc explained by SRC (R^2) is the
    shortcut-attributable share; (1 - R^2) is the residual attributed to genuine
    distribution shift (an upper bound — other unmodeled factors also land here).

    NOTE: the stronger corroboration (CSA-mask the cross-source test set and check
    accuracy selectively recovers) needs the masked-inference path and is left for the
    full Phase-4 run; this function reports the regression-based decomposition now.
    """
    coupling = couple_src_to_collapse(results_dir)
    if "delta_acc" not in coupling or coupling["delta_acc"].get("r2") != coupling["delta_acc"].get("r2"):
        return {"note": "need >=3 models with valid SRC + cross-source results first",
                "coupling": coupling}
    r2 = coupling["delta_acc"]["r2"]
    return {
        "shortcut_attributable_share": float(r2),
        "genuine_shift_residual_upper_bound": float(1.0 - r2),
        "basis": "fraction of cross-source delta_acc variance explained by SRC across models",
        "caveat": "regression-based decomposition; CSA-masking corroboration pending Phase 4",
    }
