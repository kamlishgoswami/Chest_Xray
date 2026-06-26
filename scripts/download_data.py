"""Phase 0 — download source-labeled datasets from Kaggle (PAPER_OUTLINE.md §7).

Reads configs/datasets.yaml and downloads each registered dataset into
data/raw/<source>/, recording provenance (license, access date, kaggle slug) into
data/raw/provenance.json for the TRIPOD-AI / CLAIM reporting requirement.

Prerequisites:
    pip install kaggle
    Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)
    (Get it from kaggle.com -> Account -> Create New API Token)

Usage:
    python scripts/download_data.py                  # all datasets
    python scripts/download_data.py --only tb_shenzhen tb_montgomery
    python scripts/download_data.py --list           # just print the registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "configs" / "datasets.yaml"


def load_registry():
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML not installed. Run: pip install pyyaml")
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def download_one(slug: str, dest: Path):
    """Download+unzip a Kaggle dataset into dest using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle not installed. Run: pip install kaggle  (and set up ~/.kaggle/kaggle.json)")
    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {slug} -> {dest}")
    # competitions vs datasets differ; try dataset first, fall back to competition
    try:
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    except Exception as e:  # noqa: BLE001 - surface the real cause to the user
        print(f"  dataset_download failed ({e}); trying competition API for '{slug}'")
        comp = slug.split("/")[-1]
        api.competition_download_files(comp, path=str(dest), quiet=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", help="download only these registry keys")
    p.add_argument("--list", action="store_true", help="print registry and exit")
    args = p.parse_args()

    reg = load_registry()
    datasets = reg["datasets"]

    if args.list:
        for key, d in datasets.items():
            print(f"{key:20s} role={d['role']:12s} source={d['source']:22s} slug={d['kaggle']}")
        return

    if args.only:
        keys = args.only
    else:
        # default run skips optional (large) datasets; pull them explicitly with --only
        keys = [k for k, d in datasets.items() if not d.get("optional", False)]
        skipped = [k for k, d in datasets.items() if d.get("optional", False)]
        if skipped:
            print(f"(skipping optional datasets: {skipped} — add with --only <key>)\n")

    provenance = {}
    for key in keys:
        if key not in datasets:
            print(f"!! unknown dataset key: {key} (skipping)")
            continue
        d = datasets[key]
        dest = RAW / d["source"]
        download_one(d["kaggle"], dest)
        provenance[key] = {
            "kaggle": d["kaggle"],
            "source": d["source"],
            "role": d["role"],
            "provides": d["provides"],
            "masks": d["masks"],
            "license": d["license"],
            "access_date": dt.date.today().isoformat(),
            "dest": str(dest.relative_to(ROOT)),
        }

    RAW.mkdir(parents=True, exist_ok=True)
    prov_path = RAW / "provenance.json"
    existing = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    existing.update(provenance)
    prov_path.write_text(json.dumps(existing, indent=2))
    print(f"\nProvenance recorded -> {prov_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
