"""Phase 1 — evaluate trained models on the in-domain test split (PAPER_OUTLINE.md §4.1).

For each model with a checkpoint, runs inference on the in-domain test set, computes the
standard metric suite + calibration, writes results/<model>/metrics.json, and assembles
Table A (results/table_a_in_domain.csv).

Usage:
    python scripts/evaluate.py                       # all checkpointed models
    python scripts/evaluate.py --models densenet201
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seeding import set_global_determinism  # noqa: E402

set_global_determinism(42)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None)
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
    from src.evaluation.metrics import standard_metrics, calibration_metrics

    df = load_manifest()
    test_df = filter_rows(df, split="test", roles=["in_domain"])
    test_ds = make_dataset(test_df, batch_size=32, training=False, shuffle=False)
    y_true = np.array([CLASS_TO_IDX[c] for c in test_df["disease"]])

    results_dir = ROOT / "results"
    model_dirs = ([results_dir / m for m in args.models] if args.models
                  else [d for d in results_dir.iterdir() if d.is_dir()])

    rows = []
    for mdir in model_dirs:
        name = mdir.name
        ckpt = mdir / f"{name}_best.keras"
        if not ckpt.exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        print(f"=== evaluating {name} ===")
        model = tf.keras.models.load_model(ckpt)
        y_prob = model.predict(test_ds, verbose=0)

        metrics = {**standard_metrics(y_true, y_prob), **calibration_metrics(y_true, y_prob)}
        metrics["model"] = name
        (mdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append({k: metrics[k] for k in
                     ["model", "accuracy", "acc_ci_lo", "acc_ci_hi", "f1_m", "f1_w",
                      "roc_auc_macro", "pr_auc_macro", "kappa", "mcc", "ece"]})
        print(f"  acc={metrics['accuracy']:.3f}  f1_m={metrics['f1_m']:.3f}  ece={metrics['ece']:.3f}")

    if rows:
        table_a = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        out = results_dir / "table_a_in_domain.csv"
        table_a.to_csv(out, index=False)
        print(f"\nTable A -> {out.relative_to(ROOT)}")
        print(table_a.to_string(index=False))


if __name__ == "__main__":
    main()
