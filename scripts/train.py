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

set_global_determinism(42)   # default; overridden by --seed in main() for multi-seed runs (§4.6d)


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
    p.add_argument("--seed", type=int, default=42,
                   help="training seed (§4.6d multi-seed). seed!=42 writes results/<model>_s<seed>/")
    args = p.parse_args()

    if args.seed != 42:                       # re-seed for multi-seed runs (§4.6d)
        set_global_determinism(args.seed)

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
    suffix = "" if args.seed == 42 else f"_s{args.seed}"   # keep default dirs unchanged
    for i, name in enumerate(args.models, 1):
        out_dir = ROOT / "results" / f"{name}{suffix}"
        ckpt = out_dir / f"{name}_best.keras"
        done_marker = out_dir / f"{name}.done"       # written ONLY when training fully completes
        bdir = f"{args.backup_dir}/{name}" if args.backup_dir else None

        # --resume skips a model ONLY if it FINISHED (done marker present), NOT merely because a
        # best-checkpoint exists — ModelCheckpoint writes _best.keras after epoch 1, so the file
        # existing does NOT mean training finished. (Fixes the "stopped early -> wrongly skipped" bug.)
        if args.resume and done_marker.exists():
            print(f"\n[skip] MODEL {i}/{total}: {name} — already FINISHED (done marker present)", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        interrupted = bdir is not None and Path(bdir).exists()
        tag = "RESUMING (interrupted)" if interrupted else "TRAINING"
        print(f"\n{'#'*70}\n###  {tag} MODEL {i}/{total}:  {name.upper()}\n{'#'*70}", flush=True)

        model, base = build_model(name, num_classes=cfg["num_classes"], img_size=img_size)
        hist = train_two_phase(name, model, base, train_ds, val_ds,
                               epochs=args.epochs, ckpt_path=str(ckpt), class_weight=cw,
                               backup_dir=bdir)
        # persist history (json-serializable)
        hist_json = {ph: {k: [float(v) for v in vals] for k, vals in h.history.items()}
                     for ph, h in hist.items()}
        (out_dir / f"{name}_history.json").write_text(json.dumps(hist_json, indent=2))
        done_marker.write_text("ok")                 # mark FINISHED so --resume can safely skip next time
        if bdir and Path(bdir).exists():              # finished -> drop the per-epoch backup (no longer needed)
            import shutil; shutil.rmtree(bdir, ignore_errors=True)
        print(f"\n✅ MODEL {i}/{total} DONE: {name} — best weights saved to {ckpt}", flush=True)
        print(f"   (results/ symlinked to Drive => already safe on Drive)\n", flush=True)


if __name__ == "__main__":
    main()
