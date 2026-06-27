"""Package data/raw/ into per-source zips for Google Drive staging (Colab workflow).

WHY: Colab sessions are ephemeral and Google Drive is slow with many small files
(~60k images here). Best practice: store the data as a FEW zips on Drive (durable,
fast to copy), then unzip to Colab's local /content disk at session start.

Usage (local Mac):
    python scripts/package_for_drive.py                # zip all sources -> data/drive_staging/
    python scripts/package_for_drive.py --out ~/Desktop/cxr_drive

Then upload data/drive_staging/*.zip + data/manifest.csv to a Drive folder, e.g.
    MyDrive/cxr_data/
Then open notebooks/colab_small.ipynb (or colab_full.ipynb) in Colab — they restore + run everything.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "drive_staging"),
                   help="directory to write zips into")
    args = p.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    sources = [d for d in RAW.iterdir() if d.is_dir()]
    if not sources:
        sys.exit("No source dirs under data/raw/. Run scripts/download_data.py first.")

    print(f"packaging {len(sources)} source(s) -> {out}")
    for src in sources:
        # zip name = sanitized source dir (avoid '+' / spaces in Drive)
        safe = src.name.replace("+", "_").replace(" ", "_")
        archive = out / safe
        print(f"  zipping {src.name} ...")
        shutil.make_archive(str(archive), "zip", root_dir=RAW, base_dir=src.name)
        print(f"    -> {archive.with_suffix('.zip').name}  "
              f"({(archive.with_suffix('.zip').stat().st_size / 1e9):.2f} GB)")

    # also copy the manifest + provenance for convenience
    for f in ["manifest.csv"]:
        srcf = ROOT / "data" / f
        if srcf.exists():
            shutil.copy2(srcf, out / f)
    prov = RAW / "provenance.json"
    if prov.exists():
        shutil.copy2(prov, out / "provenance.json")

    print(f"\nDone. Upload the contents of {out} to a Google Drive folder (e.g. MyDrive/cxr_data/).")
    print("Then in Colab run the bootstrap to unzip into /content (fast local disk).")


if __name__ == "__main__":
    main()
