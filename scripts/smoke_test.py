"""End-to-end SMOKE TEST on a tiny REAL subset (PAPER_OUTLINE.md §8b, local Mac).

Purpose: prove the WHOLE pipeline runs and emits well-formed output, fast, on the
laptop — BEFORE spending GPU time on Colab. It does NOT prove the science works
(tiny models are near-random; SRC<->collapse coupling is meaningless at this scale).
It proves the CODE is correct and the file contracts line up.

Strategy (no code duplication): sample a tiny manifest from the real data/manifest.csv,
swap it in temporarily, run the real train -> evaluate -> CSA/SRC -> cross-source ->
C3 pipeline through the real modules, then restore the full manifest.

Uses real images (so border/background/source structure is genuine), just very few of
them. Default: a few images per (disease, role) cell, 1 epoch, 2 small models.

Usage:
    python scripts/smoke_test.py                 # default tiny run
    python scripts/smoke_test.py --per-cell 8 --epochs 1 --models lenet5 mobilenetv3large
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _zoo_load(ckpt):
    """Load a checkpoint via the zoo loader (registers ViT custom layers first)."""
    from src.models.zoo import load_model as _lm
    return _lm(str(ckpt))

MANIFEST = ROOT / "data" / "manifest.csv"
BACKUP = ROOT / "data" / "manifest.full.bak.csv"


def make_tiny_manifest(per_cell, seed=42):
    """Sample up to `per_cell` rows per (disease, source, role, split) cell from the real manifest."""
    import pandas as pd
    df = pd.read_csv(MANIFEST)
    keys = ["disease", "source", "role", "split"]
    tiny = (df.groupby(keys, group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_cell), random_state=seed)))
    return tiny.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-cell", type=int, default=6, help="images per (disease,source,role,split) cell")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--models", nargs="+", default=["lenet5", "mobilenetv3large"])
    p.add_argument("--keep-tiny", action="store_true", help="leave the tiny manifest in place (debug)")
    args = p.parse_args()

    if not MANIFEST.exists():
        sys.exit("data/manifest.csv missing — run scripts/build_manifest.py first.")

    print(f"[smoke] backing up full manifest -> {BACKUP.name}")
    shutil.copy(MANIFEST, BACKUP)
    tiny = make_tiny_manifest(args.per_cell)
    tiny.to_csv(MANIFEST, index=False)
    print(f"[smoke] tiny manifest: {len(tiny)} rows across "
          f"{tiny['disease'].nunique()} classes, {tiny['source'].nunique()} sources")

    try:
        import numpy as np
        import tensorflow as tf
        from src.utils.seeding import set_global_determinism
        from src.data.loaders import load_manifest, filter_rows, make_dataset, CLASS_TO_IDX
        from src.shortcut import csa
        from src.shortcut.src_certificate import emit_certificate
        from src.shortcut import cross_domain
        set_global_determinism(42)

        # --- 1. TRAIN (real trainer, real images, 1 epoch) ---
        print("\n[smoke] === 1. train ===")
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"),
                            "--models", *args.models, "--epochs", str(args.epochs),
                            "--batch-size", "4"], cwd=ROOT)
        if r.returncode != 0:
            sys.exit("[smoke] FAIL: training crashed")

        # --- 2. EVALUATE (in-domain metrics + ECE) ---
        print("\n[smoke] === 2. evaluate (in-domain) ===")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                            "--models", *args.models], cwd=ROOT)
        if r.returncode != 0:
            sys.exit("[smoke] FAIL: evaluate crashed")

        # --- 3. CSA audit + SRC certificate per model ---
        print("\n[smoke] === 3. CSA audit + SRC ===")
        df = load_manifest()
        audit_df = filter_rows(df, split="test", roles=["in_domain"])
        if len(audit_df) == 0:
            audit_df = filter_rows(df, roles=["in_domain"]).head(args.per_cell * 4)
        ds = make_dataset(audit_df, batch_size=4, training=False, shuffle=False)
        images = np.concatenate([b[0].numpy() for b in ds], axis=0)
        y_true = np.array([CLASS_TO_IDX[c] for c in audit_df["disease"]])[:len(images)]

        for name in args.models:
            ckpt = ROOT / "results" / name / f"{name}_best.keras"
            if not ckpt.exists():
                print(f"[smoke] WARN: no checkpoint for {name}, skipping audit")
                continue
            model = _zoo_load(ckpt)
            # small n_boot for speed; masks=None -> CSA uses geometric fallbacks
            audit = {ch: csa.causal_effect(model, images, y_true, ch, masks=None, n_boot=50)
                     for ch in csa.ALL_CHANNELS}
            cert = emit_certificate(name, audit, ROOT / "results" / name / "certificate.json")
            print(f"[smoke] {name}: SRC={cert['src']:.3f} valid={cert['valid']} "
                  f"dominant={max(cert['per_channel'], key=cert['per_channel'].get)}")

        # --- 4. cross-source matrix ---
        print("\n[smoke] === 4. cross-source matrix ===")
        cross_domain.run_cross_source_matrix(models=args.models, batch_size=4)

        # --- 5. C3 coupling (will note <3 models if applicable) ---
        print("\n[smoke] === 5. C3 coupling ===")
        cross_domain.couple_src_to_collapse()

        print("\n[smoke] ✅ PIPELINE RAN END-TO-END. Output files written under results/.")
        print("[smoke] NOTE: numbers are meaningless at this scale — this only proves the code is correct.")

    finally:
        if not args.keep_tiny:
            print(f"\n[smoke] restoring full manifest from {BACKUP.name}")
            shutil.move(BACKUP, MANIFEST)
        else:
            print(f"[smoke] --keep-tiny set: tiny manifest left in place; full backup at {BACKUP.name}")


if __name__ == "__main__":
    main()
