"""Phase 1 — train the model zoo on the in-domain split (PAPER_OUTLINE.md §10, §4.1).

Seeds determinism, loads the manifest, builds in-domain train/val datasets, and runs
the two-phase trainer per model. Checkpoints + history go to results/<model>/.

Usage:
    python scripts/train.py                          # all models, config epochs
    python scripts/train.py --models densenet50 --epochs 5    # quick test
    python scripts/train.py --resume                 # skip models already trained
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# determinism BEFORE importing tensorflow (via the modules below)
from src.utils.seeding import set_global_determinism  # noqa: E402

set_global_determinism(42)


def load_cfg():
    import yaml
    with open(ROOT / "configs" / "train.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=cfg["models"])
    p.add_argument("--epochs", type=int, default=cfg["epochs"])
    p.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    p.add_argument("--resume", action="store_true", help="skip models with an existing checkpoint")
    p.add_argument("--backup-dir", default=None,
                   help="per-epoch BackupAndRestore dir (point at Google Drive for crash recovery)")
    args = p.parse_args()

    from src.data.loaders import load_manifest, filter_rows, make_dataset, labels_int
    from src.models.zoo import build_model
    from src.training.trainer import train_two_phase, class_weights_from_labels

    df = load_manifest()
    img_size = tuple(cfg["img_size"])

    # IN-DOMAIN only for Phase 1 (cross-source is held out for Phase 3)
    train_df = filter_rows(df, split="train", roles=["in_domain"])
    val_df = filter_rows(df, split="val", roles=["in_domain"])
    print(f"train={len(train_df)}  val={len(val_df)}  classes seen={sorted(train_df['disease'].unique())}")

    train_ds = make_dataset(train_df, img_size=img_size, batch_size=args.batch_size,
                            training=True, augment=cfg["augment"])
    val_ds = make_dataset(val_df, img_size=img_size, batch_size=args.batch_size, training=False)
    cw = class_weights_from_labels(labels_int(train_df))

    total = len(args.models)
    for i, name in enumerate(args.models, 1):
        out_dir = ROOT / "results" / name
        ckpt = out_dir / f"{name}_best.keras"
        if args.resume and ckpt.exists():
            print(f"\n[skip] MODEL {i}/{total}: {name} — already trained (checkpoint on disk)", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#'*70}\n###  TRAINING MODEL {i}/{total}:  {name.upper()}\n{'#'*70}", flush=True)
        model, base = build_model(name, num_classes=cfg["num_classes"], img_size=img_size)
        bdir = f"{args.backup_dir}/{name}" if args.backup_dir else None
        hist = train_two_phase(name, model, base, train_ds, val_ds,
                               epochs=args.epochs, ckpt_path=str(ckpt), class_weight=cw,
                               backup_dir=bdir)
        # persist history (json-serializable)
        hist_json = {ph: {k: [float(v) for v in vals] for k, vals in h.history.items()}
                     for ph, h in hist.items()}
        (out_dir / f"{name}_history.json").write_text(json.dumps(hist_json, indent=2))
        print(f"\n✅ MODEL {i}/{total} DONE: {name} — best weights saved to {ckpt}", flush=True)
        print(f"   (if results/ is symlinked to Drive, this is already safe on Drive)\n", flush=True)


if __name__ == "__main__":
    main()
