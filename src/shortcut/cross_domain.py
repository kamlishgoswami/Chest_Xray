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


def _zoo_load(ckpt):
    """Load a checkpoint via the zoo loader (registers ViT custom layers first)."""
    from src.models.zoo import load_model as _lm
    return _lm(str(ckpt))


def _predict(model, df, batch_size=32):
    """Return (y_true, y_prob) for a manifest subset, or (None, None) if empty."""
    from src.data.loaders import make_dataset, CLASS_TO_IDX
    if len(df) == 0:
        return None, None
    ds = make_dataset(df, batch_size=batch_size, training=False, shuffle=False)
    y_true = np.array([CLASS_TO_IDX[c] for c in df["disease"]])
    y_prob = model.predict(ds, verbose=0)
    return y_true, y_prob


def _metrics_on(model, df, batch_size=32):
    """Run a model over the rows in `df`; return accuracy, ece, n, classes_present, + y_true/y_prob."""
    from src.evaluation.metrics import standard_metrics, calibration_metrics

    y_true, y_prob = _predict(model, df, batch_size)
    if y_true is None:
        return {"accuracy": float("nan"), "ece": float("nan"), "n": 0, "classes": [],
                "y_true": None, "y_prob": None}

    m = {**standard_metrics(y_true, y_prob), **calibration_metrics(y_true, y_prob)}
    return {
        "accuracy": float(m["accuracy"]),
        "ece": float(m["ece"]),
        "n": int(len(df)),
        "classes": sorted(df["disease"].unique().tolist()),
        "y_true": y_true, "y_prob": y_prob,
    }


def _ece_after_temperature(val_true, val_prob, cross_true, cross_prob):
    """§4.6c — ECE on cross-source AFTER temperature scaling fit on the in-domain val set.

    Models output softmax probs; we recover logits as log(prob) (temperature scaling is invariant
    to an additive constant, so log-prob suffices), fit T on val by minimizing NLL, apply to
    cross-source, and report the post-T ECE. If SRC still predicts THIS, miscalibration is not
    merely global over/under-confidence that T fixes.
    """
    from src.evaluation.metrics import fit_temperature, expected_calibration_error
    if val_true is None or cross_true is None:
        return float("nan")
    val_logits = np.log(np.clip(val_prob, 1e-12, 1.0))
    T = fit_temperature(val_true, val_logits)
    z = np.log(np.clip(cross_prob, 1e-12, 1.0)) / max(T, 1e-6)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return float(expected_calibration_error(cross_true, p))


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
    # val set (for temperature fitting, §4.6c) — restricted to the same classes
    val_df = filter_rows(df, split="val", roles=["in_domain"])
    val_df = val_df[val_df["disease"].isin(cross_classes)]

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
        model = _zoo_load(ckpt)

        indom = _metrics_on(model, indom_df, batch_size)
        cross = _metrics_on(model, cross_df, batch_size)
        val_true, val_prob = _predict(model, val_df, batch_size)
        # §4.6c: cross-source ECE AFTER temperature scaling fit on val
        cross_ece_post_ts = _ece_after_temperature(val_true, val_prob,
                                                    cross["y_true"], cross["y_prob"])

        rec = {
            "model": name,
            "cross_source_classes": cross_classes,
            "in_domain_acc": indom["accuracy"], "in_domain_ece": indom["ece"], "in_domain_n": indom["n"],
            "cross_source_acc": cross["accuracy"], "cross_source_ece": cross["ece"], "cross_source_n": cross["n"],
            "cross_source_ece_post_ts": cross_ece_post_ts,
            "delta_acc": float(indom["accuracy"] - cross["accuracy"]),
            "delta_ece": float(cross["ece"] - indom["ece"]),
            "delta_ece_post_ts": float(cross_ece_post_ts - indom["ece"]),
        }
        rows.append(rec)
        print(f"  in-domain acc={indom['accuracy']:.3f} ece={indom['ece']:.3f} | "
              f"cross acc={cross['accuracy']:.3f} ece={cross['ece']:.3f} "
              f"(post-TS {cross_ece_post_ts:.3f}) | "
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

    pairs = [(cert_src.get(r["model"]), r["delta_acc"], r["delta_ece"],
              r.get("delta_ece_post_ts", float("nan")), r["model"])
             for r in cs if cert_src.get(r["model"]) is not None]

    out = {"n_models": len(pairs),
           "pairs": [{"model": m, "src": s, "delta_acc": da, "delta_ece": de,
                      "delta_ece_post_ts": dp}
                     for (s, da, de, dp, m) in pairs]}

    if len(pairs) >= 3:
        src = np.array([p[0] for p in pairs], float)
        for dep_name, idx in (("delta_acc", 1), ("delta_ece", 2), ("delta_ece_post_ts", 3)):
            y = np.array([p[idx] for p in pairs], float)
            if not np.isnan(y).all():
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


def _baseline_predictors(results_dir):
    """Per-model trivial predictors for §4.6a: in-domain acc, mean confidence, entropy, ECE,
    Grad-CAM out-of-lung fraction. Read from existing artifacts (no recompute)."""
    cs = json.loads((Path(results_dir) / "cross_source.json").read_text())
    preds = {}
    for r in cs:
        m = r["model"]
        d = {"in_domain_acc": r.get("in_domain_acc"), "in_domain_ece": r.get("in_domain_ece")}
        metrics_p = Path(results_dir) / m / "metrics.json"
        if metrics_p.exists():
            mj = json.loads(metrics_p.read_text())
            # mean confidence / entropy if predictions were stored; else fall back to acc/ece only
            d["roc_auc_macro"] = mj.get("roc_auc_macro")
        xai_p = Path(results_dir) / m / "xai.json"
        if xai_p.exists():
            xj = json.loads(xai_p.read_text())
            gc = xj.get("grad_cam", {})
            il = gc.get("in_lung")
            d["out_of_lung_fraction"] = (1.0 - il) if (il is not None and il == il) else None
        preds[m] = d
    return preds


def src_vs_baselines(results_dir=None):
    """§4.6a — head-to-head R^2: does SRC predict collapse BETTER than trivial predictors?

    For each dependent (delta_acc, delta_ece, delta_ece_post_ts) regress it on SRC and on each
    baseline (in-domain acc, in-domain ECE, out-of-lung saliency fraction), reporting R^2 for all.
    Establishes that SRC adds predictive value beyond freely-available alternatives.
    """
    results_dir = Path(results_dir) if results_dir else (ROOT / "results")
    cs = json.loads((results_dir / "cross_source.json").read_text())
    cert_src = _load_src_scores([results_dir / r["model"] for r in cs], results_dir)
    base = _baseline_predictors(results_dir)

    deps = ["delta_acc", "delta_ece", "delta_ece_post_ts"]
    predictors = {"SRC": {r["model"]: cert_src.get(r["model"]) for r in cs}}
    for key in ["in_domain_acc", "in_domain_ece", "out_of_lung_fraction"]:
        predictors[key] = {m: base.get(m, {}).get(key) for m in (r["model"] for r in cs)}

    dep_vals = {d: {r["model"]: r.get(d) for r in cs} for d in deps}
    models = [r["model"] for r in cs]

    out = {"n_models": len(models), "comparison": {}}
    for d in deps:
        out["comparison"][d] = {}
        for pname, pmap in predictors.items():
            x = np.array([pmap.get(m) for m in models], float)
            y = np.array([dep_vals[d].get(m) for m in models], float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 3:
                out["comparison"][d][pname] = _ols_fit(x[ok], y[ok])
            else:
                out["comparison"][d][pname] = {"note": f"<3 valid points ({int(ok.sum())})"}
    (results_dir / "src_vs_baselines.json").write_text(json.dumps(out, indent=2))
    print("[baselines] R^2 head-to-head ->", {d: {p: round(v.get("r2", float("nan")), 3)
          for p, v in out["comparison"][d].items()} for d in deps})
    return out


def lomo_prediction(results_dir=None):
    """§4.6b — Leave-One-Model-Out out-of-sample prediction (earns the word "predicts").

    For each model k: fit SRC->dep OLS on the OTHER models, predict dep_k, collect predicted-vs-
    actual. Reports LOMO R^2 and MAE per dependent. This is genuine out-of-sample validation, not
    the in-sample fit of couple_src_to_collapse(). Needs >=4 models to be meaningful (3 to fit + 1 held out).
    """
    results_dir = Path(results_dir) if results_dir else (ROOT / "results")
    cs = json.loads((results_dir / "cross_source.json").read_text())
    cert_src = _load_src_scores([results_dir / r["model"] for r in cs], results_dir)
    models = [r["model"] for r in cs if cert_src.get(r["model"]) is not None]
    src = {m: cert_src[m] for m in models}

    out = {"n_models": len(models)}
    for dep in ["delta_acc", "delta_ece", "delta_ece_post_ts"]:
        y = {r["model"]: r.get(dep) for r in cs}
        pts = [(src[m], y[m], m) for m in models if y.get(m) is not None and y[m] == y[m]]
        if len(pts) < 4:
            out[dep] = {"note": f"need >=4 models for LOMO ({len(pts)} available)"}
            continue
        preds, actuals = [], []
        for i in range(len(pts)):
            train = [pts[j] for j in range(len(pts)) if j != i]
            xt = np.array([p[0] for p in train]); yt = np.array([p[1] for p in train])
            if np.allclose(xt.std(), 0):
                continue
            slope, intercept = np.polyfit(xt, yt, 1)
            preds.append(slope * pts[i][0] + intercept)
            actuals.append(pts[i][1])
        preds, actuals = np.array(preds), np.array(actuals)
        ss_res = float(((actuals - preds) ** 2).sum())
        ss_tot = float(((actuals - actuals.mean()) ** 2).sum())
        out[dep] = {
            "lomo_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "lomo_mae": float(np.abs(actuals - preds).mean()),
            "predicted_vs_actual": [{"model": pts[i][2], "predicted": float(preds[i]),
                                     "actual": float(actuals[i])} for i in range(len(preds))],
        }
    (results_dir / "lomo.json").write_text(json.dumps(out, indent=2))
    print("[lomo] out-of-sample R^2 ->",
          {d: round(out[d].get("lomo_r2", float("nan")), 3) if isinstance(out[d], dict) and "lomo_r2" in out[d]
           else out[d].get("note") for d in ["delta_acc", "delta_ece", "delta_ece_post_ts"]})
    return out


def partial_correlation(results_dir=None):
    """§4.6 — partial correlation of SRC with collapse, CONTROLLING for in-domain accuracy.

    Rules out "SRC just proxies a weak in-domain model." Computes partial r via residualization:
    regress out in-domain acc from both SRC and the dependent, correlate the residuals.
    """
    results_dir = Path(results_dir) if results_dir else (ROOT / "results")
    cs = json.loads((results_dir / "cross_source.json").read_text())
    cert_src = _load_src_scores([results_dir / r["model"] for r in cs], results_dir)
    models = [r["model"] for r in cs if cert_src.get(r["model"]) is not None]

    src = np.array([cert_src[m] for m in models], float)
    ctrl = np.array([next(r["in_domain_acc"] for r in cs if r["model"] == m) for m in models], float)
    out = {"n_models": len(models), "control": "in_domain_acc"}

    def _resid(y, x):
        if np.allclose(x.std(), 0):
            return y - y.mean()
        s, b = np.polyfit(x, y, 1)
        return y - (s * x + b)

    if len(models) >= 4 and not np.allclose(ctrl.std(), 0):
        rs = _resid(src, ctrl)
        for dep in ["delta_acc", "delta_ece", "delta_ece_post_ts"]:
            y = np.array([next((r.get(dep) for r in cs if r["model"] == m), np.nan) for m in models], float)
            if np.isnan(y).any():
                out[dep] = {"note": "missing dependent values"}; continue
            ry = _resid(y, ctrl)
            out[dep] = {"partial_r": float(np.corrcoef(rs, ry)[0, 1]) if rs.std() and ry.std() else float("nan")}
    else:
        out["note"] = "need >=4 models with varying in-domain accuracy"
    (results_dir / "partial_correlation.json").write_text(json.dumps(out, indent=2))
    print("[partial-corr] ->", {k: v for k, v in out.items() if k.startswith("delta")})
    return out


def csa_mask_recovery(models=None, batch_size=32, per_class=200, results_dir=None):
    """§4.7 (real corroboration) — CSA-mask the CROSS-SOURCE test set; does accuracy RECOVER?

    If the cross-source collapse was driven by shortcut reliance, then neutralizing the shortcut
    channels (border+background+source_signature) on the cross-source images should RECOVER accuracy.
    If collapse was genuine distribution shift, masking shortcuts should NOT help (or hurt).

    Per model: cross-source accuracy raw vs after masking each shortcut channel. recovery>0 supports
    a shortcut-driven collapse. Writes results/csa_mask_recovery.json.
    """
    import tensorflow as tf
    from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
    from . import csa

    results_dir = Path(results_dir) if results_dir else (ROOT / "results")
    df = load_manifest()
    cross_df = filter_rows(df, roles=["cross_source"])
    cross_df = cross_df.groupby("disease", group_keys=False)[cross_df.columns.tolist()].apply(
        lambda g: g.sample(min(len(g), per_class), random_state=42))
    ds = make_dataset(cross_df, batch_size=batch_size, training=False, shuffle=False)
    images = np.concatenate([b[0].numpy() for b in ds], axis=0)
    y_true = np.array([CLASS_TO_IDX[c] for c in cross_df["disease"]])[:len(images)]

    model_dirs = ([results_dir / m for m in models] if models
                  else [d for d in results_dir.iterdir() if d.is_dir()])
    out = {"n_cross_images": int(len(images)), "models": {}}
    for mdir in model_dirs:
        name = mdir.name
        ckpt = mdir / f"{name}_best.keras"
        if not ckpt.exists():
            continue
        model = _zoo_load(ckpt)
        raw_acc = float((model.predict(images, verbose=0).argmax(1) == y_true).mean())
        rec = {"cross_source_acc_raw": raw_acc, "per_channel_recovery": {}}
        for ch in csa.SHORTCUT_CHANNELS:
            masked = np.stack([csa.intervene(im, ch, None) for im in images])
            acc = float((model.predict(masked, verbose=0).argmax(1) == y_true).mean())
            rec["per_channel_recovery"][ch] = {"acc_after_mask": acc, "recovery": acc - raw_acc}
        out["models"][name] = rec
        print(f"[csa-recovery] {name}: raw={raw_acc:.3f} "
              f"recovery={ {c: round(v['recovery'],3) for c,v in rec['per_channel_recovery'].items()} }",
              flush=True)
    (results_dir / "csa_mask_recovery.json").write_text(json.dumps(out, indent=2))
    return out


def confounder_separation(results_dir=None, run_mask_recovery=True, models=None):
    """§4.7 — split cross-source collapse into shortcut-attributable vs genuine-shift residual.

    Two complementary pieces:
    (1) regression decomposition: R^2 of SRC->delta_acc = shortcut-attributable share;
    (2) CSA-mask recovery (csa_mask_recovery): does masking shortcuts on cross-source recover acc?
    """
    coupling = couple_src_to_collapse(results_dir)
    out = {}
    if "delta_acc" in coupling and coupling["delta_acc"].get("r2") == coupling["delta_acc"].get("r2"):
        r2 = coupling["delta_acc"]["r2"]
        out["regression_decomposition"] = {
            "shortcut_attributable_share": float(r2),
            "genuine_shift_residual_upper_bound": float(1.0 - r2),
            "basis": "fraction of cross-source delta_acc variance explained by SRC across models",
        }
    else:
        out["regression_decomposition"] = {"note": "need >=3 models with valid SRC + cross-source"}

    if run_mask_recovery:
        try:
            out["csa_mask_recovery"] = csa_mask_recovery(models=models, results_dir=results_dir)
        except Exception as e:                    # don't let an inference failure kill the stage
            out["csa_mask_recovery"] = {"note": f"skipped: {type(e).__name__}: {e}"}

    (Path(results_dir or (ROOT / "results")) / "confounder_separation.json").write_text(json.dumps(out, indent=2))
    return out
