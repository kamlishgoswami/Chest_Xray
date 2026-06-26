"""Phase 0 — build the source-labeled manifest from downloaded data (PAPER_OUTLINE.md §3.1).

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --no-phash      # skip dedup (faster smoke test)

Writes data/manifest.csv and prints a class x source x split summary so you can
sanity-check the cross-source design before training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_manifest, load_registry  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-phash", action="store_true", help="skip perceptual-hash dedup")
    args = p.parse_args()

    registry = load_registry()
    df = build_manifest(registry, compute_phash=not args.no_phash)

    out = ROOT / "data" / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"\nmanifest -> {out.relative_to(ROOT)}  ({len(df)} images)")

    print("\n=== class x source (counts) ===")
    print(df.groupby(["disease", "source"]).size().unstack(fill_value=0))
    print("\n=== split x role ===")
    print(df.groupby(["role", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
