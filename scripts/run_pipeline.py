"""Unified pipeline orchestrator — P1..P9 (PAPER_OUTLINE.md §4z).

ONE entrypoint for: the local smoke test AND both Colab scripts (one codebase, three configs).
Each stage is independent and idempotent; choose stages with --stages.

Stages:
  train      P1  train the model zoo                 -> results/<m>/<m>_best.keras
  eval       P2  in-domain metrics + calibration     -> results/<m>/metrics.json, table_a_in_domain.csv
  audit      P3  CSA audit + SRC certificate         -> results/<m>/certificate.json
  cross      P4  cross-source collapse + C3 coupling -> results/cross_source.json, c3_coupling.json
  xai        P5  Grad-CAM/IG faithfulness + in-lung  -> results/<m>/xai.json
  robust     P6  SSP robustness                      -> results/<m>/robustness.json
  abstain    P7  accuracy/coverage curves            -> results/<m>/abstention.json
  stats      P9  Friedman/McNemar/Holm               -> results/stats_summary.json
  report     P8  figures + LaTeX tables              -> results/figures/, results/tables/

Usage:
  python scripts/run_pipeline.py --models densenet201 resnet50 vit --epochs 20
  python scripts/run_pipeline.py --stages audit cross report          # skip training
  python scripts/run_pipeline.py --smoke                              # tiny-subset end-to-end
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

# quiet, clean logs for Colab (cosmetic only — no behavior change)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_STAGES = ["train", "eval", "audit", "cross", "xai", "robust", "abstain", "stats", "report"]


def _banner(stage, i, total):
    """Visible progress marker so the notebook always shows where the run is."""
    print(f"\n{'='*64}\n[{i}/{total}] STAGE: {stage}\n{'='*64}", flush=True)


def _stream(cmd):
    """Run a subprocess with LIVE, unbuffered output into the notebook (no buffering)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _audit_images(per_class=300):
    """Sample an audit set from the in-domain test split. Returns (images, y_true, masks)."""
    import numpy as np
    from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
    df = load_manifest()
    adf = filter_rows(df, split="test", roles=["in_domain"])
    if len(adf) == 0:
        adf = filter_rows(df, roles=["in_domain"])
    adf = adf.groupby("disease", group_keys=False)[adf.columns.tolist()].apply(
        lambda g: g.sample(min(len(g), per_class), random_state=42))
    ds = make_dataset(adf, batch_size=32, training=False, shuffle=False)
    images = np.concatenate([b[0].numpy() for b in ds], axis=0)
    y_true = np.array([CLASS_TO_IDX[c] for c in adf["disease"]])[:len(images)]
    return images, y_true, [None] * len(images)   # masks=None -> CSA geometric fallback


def stage_audit(models, per_class):
    import tensorflow as tf
    from src.shortcut import csa
    from src.shortcut.src_certificate import emit_certificate
    images, y_true, masks = _audit_images(per_class)
    for i, m in enumerate(models, 1):
        ckpt = ROOT / "results" / m / f"{m}_best.keras"
        if not ckpt.exists():
            print(f"[audit] no checkpoint {m}; skip", flush=True); continue
        print(f"[audit] ({i}/{len(models)}) {m} ...", flush=True)
        model = tf.keras.models.load_model(ckpt)
        audit = {ch: csa.causal_effect(model, images, y_true, ch, masks=masks, n_boot=1000)
                 for ch in csa.ALL_CHANNELS}
        cert = emit_certificate(m, audit, ROOT / "results" / m / "certificate.json")
        print(f"[audit] {m}: SRC={cert['src']:.3f} valid={cert['valid']}", flush=True)


def stage_xai(models, per_class):
    import tensorflow as tf
    from src.xai import explain
    images, y_true, masks = _audit_images(min(per_class, 60))   # XAI is per-image, keep small
    n_img = len(images)
    for mi, m in enumerate(models, 1):
        ckpt = ROOT / "results" / m / f"{m}_best.keras"
        if not ckpt.exists():
            print(f"[xai] no checkpoint {m}; skip", flush=True); continue
        print(f"[xai] ({mi}/{len(models)}) {m}: {n_img} images × {len(explain.SALIENCY_METHODS)} methods ...",
              flush=True)
        model = tf.keras.models.load_model(ckpt)
        agg = {meth: {"deletion_auc": [], "insertion_auc": [], "in_lung": []}
               for meth in explain.SALIENCY_METHODS}
        for j, (img, y, mask) in enumerate(zip(images, y_true, masks), 1):
            for meth, fn in explain.SALIENCY_METHODS.items():
                sal = fn(model, img, int(y))
                di = explain.deletion_insertion_auc(model, img, int(y), sal, steps=12)
                agg[meth]["deletion_auc"].append(di["deletion_auc"])
                agg[meth]["insertion_auc"].append(di["insertion_auc"])
                agg[meth]["in_lung"].append(explain.in_lung_fraction(sal, mask))
            if j % 10 == 0 or j == n_img:
                print(f"        [xai] {m}: {j}/{n_img} images", flush=True)
        import numpy as np
        out = {meth: {k: float(np.nanmean(v)) for k, v in d.items()} for meth, d in agg.items()}
        (ROOT / "results" / m / "xai.json").write_text(json.dumps(out, indent=2))
        print(f"[xai] {m}: {out}", flush=True)


def stage_robust(models, per_class):
    import tensorflow as tf
    from src.robustness import perturbations
    images, y_true, _ = _audit_images(min(per_class, 100))
    for m in models:
        ckpt = ROOT / "results" / m / f"{m}_best.keras"
        if not ckpt.exists():
            print(f"[robust] no checkpoint {m}; skip"); continue
        model = tf.keras.models.load_model(ckpt)
        print(f"[robust] {m}: 7 perturbation families ...", flush=True)
        res = perturbations.evaluate_robustness(model, images, y_true)
        (ROOT / "results" / m / "robustness.json").write_text(json.dumps(res, indent=2))
        print(f"[robust] {m}: clean={res['clean_accuracy']:.3f} MRR={res['mrr']:.3f}", flush=True)


def stage_abstain(models):
    import tensorflow as tf, numpy as np
    from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
    from src.shortcut.abstention import accuracy_coverage_curve, coverage_at_target_accuracy
    df = load_manifest()
    tdf = filter_rows(df, split="test", roles=["in_domain"])
    ds = make_dataset(tdf, batch_size=32, training=False, shuffle=False)
    y_true = np.array([CLASS_TO_IDX[c] for c in tdf["disease"]])
    for m in models:
        ckpt = ROOT / "results" / m / f"{m}_best.keras"
        if not ckpt.exists():
            print(f"[abstain] no checkpoint {m}; skip"); continue
        model = tf.keras.models.load_model(ckpt)
        y_prob = model.predict(ds, verbose=0)
        curve = accuracy_coverage_curve(y_true[:len(y_prob)], y_prob)
        curve["coverage_at_95"] = coverage_at_target_accuracy(curve, 0.95)
        (ROOT / "results" / m / "abstention.json").write_text(json.dumps(curve, indent=2))
        print(f"[abstain] {m}: AURC={curve['aurc']:.3f} cov@95={curve['coverage_at_95']:.2f}")


def stage_stats(models):
    import tensorflow as tf, numpy as np
    from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
    from src.evaluation.metrics import cross_model_stats
    df = load_manifest()
    tdf = filter_rows(df, split="test", roles=["in_domain"])
    ds = make_dataset(tdf, batch_size=32, training=False, shuffle=False)
    y_true = np.array([CLASS_TO_IDX[c] for c in tdf["disease"]])
    correct = {}
    for m in models:
        ckpt = ROOT / "results" / m / f"{m}_best.keras"
        if not ckpt.exists():
            continue
        model = tf.keras.models.load_model(ckpt)
        pred = model.predict(ds, verbose=0).argmax(1)
        correct[m] = (pred == y_true[:len(pred)]).astype(int)
    stats = cross_model_stats(correct)
    (ROOT / "results" / "stats_summary.json").write_text(json.dumps(stats, indent=2))
    print(f"[stats] {stats.get('friedman', stats.get('note'))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--per-class", type=int, default=300, help="audit/xai/robust sample cap per class")
    p.add_argument("--backup-dir", default=None,
                   help="per-epoch BackupAndRestore dir for training (point at Drive on Colab)")
    p.add_argument("--smoke", action="store_true", help="tiny-subset end-to-end (delegates to smoke_test.py)")
    args = p.parse_args()

    if args.smoke:
        sys.exit(subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py")]).returncode)

    from src.data.loaders import load_manifest
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    models = args.models or cfg["models"]

    stages = args.stages
    total = len(stages)
    done = 0

    if "train" in stages:
        done += 1; _banner("train (live Keras progress below)", done, total)
        train_cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
                     "--models", *models, "--epochs", str(args.epochs),
                     "--batch-size", str(args.batch_size), "--resume"]
        if args.backup_dir:
            train_cmd += ["--backup-dir", args.backup_dir]
        _stream(train_cmd)
    if "eval" in stages:
        done += 1; _banner("eval (in-domain metrics + calibration)", done, total)
        _stream([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--models", *models])
    if "audit" in stages:
        done += 1; _banner("audit (CSA + SRC certificate)", done, total)
        stage_audit(models, args.per_class)
    if "cross" in stages:
        done += 1; _banner("cross (cross-source collapse + C3 coupling)", done, total)
        from src.shortcut import cross_domain
        cross_domain.run_cross_source_matrix(models=models, batch_size=args.batch_size)
        cross_domain.couple_src_to_collapse()
    if "xai" in stages:
        done += 1; _banner("xai (Grad-CAM / IG faithfulness + in-lung)", done, total)
        stage_xai(models, args.per_class)
    if "robust" in stages:
        done += 1; _banner("robust (SSP perturbations)", done, total)
        stage_robust(models, args.per_class)
    if "abstain" in stages:
        done += 1; _banner("abstain (accuracy/coverage curves)", done, total)
        stage_abstain(models)
    if "stats" in stages:
        done += 1; _banner("stats (Friedman / McNemar / Holm)", done, total)
        stage_stats(models)
    if "report" in stages:
        done += 1; _banner("report (figures + LaTeX tables)", done, total)
        from src.reporting.figures import generate_all
        generate_all()

    print("\n[run_pipeline] ✅ done:", stages, flush=True)


if __name__ == "__main__":
    main()
