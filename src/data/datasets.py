"""Phase 0 — Source-labeled manifest builder + leakage control (PAPER_OUTLINE.md §3.1, §7).

Builds a single dataframe:  [image_path, disease, source, role, patient_id, phash, split]
from the downloaded raw datasets (data/raw/<source>/...).

KEY DESIGN RULES:
  - Every image carries a SOURCE label. Cross-source == cross-domain (the shortcut shows here).
  - role="in_domain" rows are split train/val/test; role="cross_source" rows are 100% test.
  - Leakage control (§3.1.1): perceptual-hash (phash) dedup BEFORE splitting; patient-level
    splits where a patient_id is derivable from the filename, else duplicate-filtering only.

This module discovers images by walking each source dir; class-folder names are normalized
to the canonical {Covid, Normal, Pneumonia, TB}. Folder layouts vary per dataset, so the
class normalization map is intentionally permissive and logged.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DISEASE_CLASSES = ["Covid", "Normal", "Pneumonia", "TB"]

# Folder names that are NOT one of our 4 classes -> always skip (checked FIRST).
# 'non-covid' contains 'covid' as a substring, 'lung_opacity' is a 5th class we don't use.
CLASS_EXCLUDE = ("non-covid", "non_covid", "lung_opacity", "lung-opacity", "lung opacity")

# Permissive folder-name -> canonical class. Order matters: more specific aliases first.
CLASS_ALIASES = {
    "viral pneumonia": "Pneumonia",
    "covid": "Covid",          # matches 'COVID', 'COVID-19' (non-covid excluded above)
    "normal": "Normal",
    "pneumonia": "Pneumonia",
    "bacteria": "Pneumonia",
    "virus": "Pneumonia",
    "tuberculosis": "TB",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# A CXR image is excluded as a MASK only if one of its PATH COMPONENTS is exactly a mask
# folder. We match whole folder names (not substrings) so that 'Lung Segmentation Data'
# (a dataset dir full of real images under .../images/) is NOT mistaken for masks.
MASK_DIR_NAMES = {"mask", "masks", "manualmask", "manualmasks",
                  "leftmask", "rightmask", "lung_mask", "lung masks", "lungmasks"}

# junk / duplicate-nesting dir markers to skip during discovery
SKIP_DIR_MARKERS = ("__macosx",)

# COVID-QU-Ex ships the SAME images twice: 'Lung Segmentation Data' (complete canonical set,
# 11,956 COVID — matches the official count, AND ships lung masks we need) and
# 'Infection Segmentation Data' (a redundant 2,913-image subset for a task we don't use).
# Ingesting both double-counts. Keep ONLY Lung Segmentation Data.
SKIP_PATH_SUBSTRINGS = ("infection segmentation data",)


class _Excluded:
    """Sentinel: this path component is an EXPLICITLY excluded class (e.g. Lung_Opacity).
    Distinct from None (= 'not a class name') so the discovery loop STOPS and drops the image
    instead of walking further up the path and matching a parent dataset folder by substring."""


EXCLUDED = _Excluded()


def normalize_class(folder_name: str):
    """Map a raw folder/label to a canonical class, EXCLUDED sentinel, or None.

    - EXCLUDED -> this component names a class we deliberately drop (lung_opacity, non-covid).
                  The caller must DROP the image, not keep searching parent folders.
    - canonical str -> a target class.
    - None -> this component is not a class name (keep searching other path components).
    """
    s = folder_name.strip().lower()
    if any(x in s for x in CLASS_EXCLUDE):
        return EXCLUDED
    for alias, canon in CLASS_ALIASES.items():
        if alias in s:
            return canon
    return None


def derive_patient_id(path: Path) -> str | None:
    """Best-effort patient id from filename (e.g. 'person12_...', 'CHNCXR_0001_...')."""
    name = path.stem
    m = re.search(r"(person\d+|patient\d+|CHNCXR_\d+|MCUCXR_\d+)", name, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # RSNA filenames are bare UUIDs and are 1-image-per-patient -> the UUID IS the patient id
    if "RSNA" in path.parts:
        return name.lower()
    return None


def perceptual_hash(path: Path):
    """phash of an image; None if imagehash/Pillow unavailable or file unreadable."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None
    try:
        return str(imagehash.phash(Image.open(path).convert("L")))
    except Exception:  # noqa: BLE001 - skip unreadable files, don't crash the build
        return None


def _resolve_tb_bundle(img: Path):
    """For the kmader TB bundle, return (source, role, disease) from the filename.

    Shenzhen (CHNCXR_*) -> in_domain; Montgomery (MCUCXR_*) -> cross_source.
    The trailing label encodes the class: '..._0.png' = Normal, '..._1.png' = TB
    (Jaeger et al. 2014 convention). Returns None if neither source (skip).
    """
    # Decide source from the FILENAME prefix (the path contains 'Shenzhen+Montgomery',
    # which would match both — so we must not key on the path here).
    name = img.name.upper()
    if name.startswith("CHNCXR"):
        source, role = "Shenzhen", "in_domain"
    elif name.startswith("MCUCXR"):
        source, role = "Montgomery", "cross_source"
    else:
        return None
    # class from trailing _0 / _1 before the extension
    m = re.search(r"_([01])\.[a-z]+$", img.name, re.IGNORECASE)
    if m is None:
        return None  # mask files / unlabeled -> skip
    disease = "TB" if m.group(1) == "1" else "Normal"
    return source, role, disease


# RSNA: labels live in stage2_train_metadata.csv (filenames are UUIDs), NOT in folders.
# 3 classes -> we map: 'Normal'->Normal, 'Lung Opacity'->Pneumonia, and DROP
# 'No Lung Opacity / Not Normal' (abnormal-but-not-pneumonia -> would be label noise).
_RSNA_LABELS = None  # lazy-loaded {patientId: disease-or-None}


def _load_rsna_labels():
    global _RSNA_LABELS
    if _RSNA_LABELS is None:
        import csv
        _RSNA_LABELS = {}
        csv_path = RAW / "RSNA" / "stage2_train_metadata.csv"
        if csv_path.exists():
            for r in csv.DictReader(open(csv_path)):
                cls = r.get("class", "").strip()
                if cls == "Normal":
                    _RSNA_LABELS[r["patientId"]] = "Normal"
                elif cls == "Lung Opacity":
                    _RSNA_LABELS[r["patientId"]] = "Pneumonia"
                # 'No Lung Opacity / Not Normal' -> intentionally NOT added (excluded)
    return _RSNA_LABELS


def _resolve_rsna(img: Path):
    """RSNA Training/Images/<patientId>.png -> (source, role, disease) via the CSV.

    Returns None for: the unlabeled Test/ set, mask files, and the excluded 'Not Normal' class.
    """
    # only label the Training/Images set; Test/ has no public labels -> skip
    if "Test" in img.parts:
        return None
    labels = _load_rsna_labels()
    disease = labels.get(img.stem)        # filename stem == patientId UUID
    if disease is None:
        return None                        # excluded class or not in label map
    return "RSNA", "cross_source", disease


def discover_rows(registry: dict):
    """Walk data/raw/<source>/ for each registered dataset, yielding row dicts (no split yet)."""
    for d in registry["datasets"].values():
        source_dir = RAW / d["source"]
        if not source_dir.exists():
            continue
        is_tb_bundle = d["source"] == "Shenzhen+Montgomery"
        is_rsna = d["source"] == "RSNA"
        for img in source_dir.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            parts_lower = [part.lower() for part in img.parts]
            # skip macOS junk and other non-data dirs
            if any(s in p for p in parts_lower for s in SKIP_DIR_MARKERS):
                continue
            # skip redundant duplicate sub-datasets (e.g. COVID-QU-Ex Infection Segmentation Data)
            path_lower = "/".join(parts_lower)
            if any(s in path_lower for s in SKIP_PATH_SUBSTRINGS):
                continue
            # skip segmentation masks: a path COMPONENT is exactly a mask folder name
            if any(p in MASK_DIR_NAMES for p in parts_lower):
                continue
            # Kermany ships a duplicate nested copy at 'chest_xray/chest_xray/...'; skip ONLY
            # that inner copy (consecutive repeat), keeping 'chest_xray/test|train|val/...'.
            if any(parts_lower[i] == "chest_xray" and parts_lower[i + 1] == "chest_xray"
                   for i in range(len(parts_lower) - 1)):
                continue
            # phash dedup (later) removes any remaining true duplicate images across copies.
            # The kmader TB bundle has no class folders; class + source come from the
            # filename (CHNCXR_/MCUCXR_ + trailing _0/_1). Other datasets use class folders.
            if is_tb_bundle:
                resolved = _resolve_tb_bundle(img)
                if resolved is None:
                    continue
                source, role, disease = resolved
            elif is_rsna:
                # RSNA Masks/ are segmentation files (same UUID as images) -> skip via path
                if "Masks" in img.parts:
                    continue
                resolved = _resolve_rsna(img)
                if resolved is None:
                    continue
                source, role, disease = resolved
            else:
                # Walk path components from the FILE up. Stop at the first that is either a
                # class OR an explicit exclusion. Critically: an EXCLUDED component (e.g.
                # 'Lung_Opacity') DROPS the image — we must NOT keep walking up and match a
                # parent dataset folder by substring (e.g. 'COVID-19_Radiography_Dataset').
                disease = None
                for part in reversed(img.parts):
                    res = normalize_class(part)
                    if res is EXCLUDED:
                        disease = None
                        break               # excluded class -> drop, do not search higher
                    if res is not None:
                        disease = res
                        break
                if disease is None or disease not in d["provides"]:
                    continue
                source, role = d["source"], d["role"]

            yield {
                "image_path": str(img.relative_to(ROOT)),
                "disease": disease,
                "source": source,
                "role": role,
                "patient_id": derive_patient_id(img),
                "mask_path": _find_mask(img),   # real lung mask if one ships, else "" (CSA falls back)
            }


def _find_mask(img: Path) -> str:
    """Locate the paired lung mask for an image: a same-named file under a sibling
    'masks' / 'lung masks' / 'ManualMask' folder. Returns '' if none (CSA uses oval fallback).

    Layout examples:
      .../<class>/images/covid_1.png        -> .../<class>/lung masks/covid_1.png   (COVID-QU-Ex)
      .../<class>/images/COVID-1.png        -> .../<class>/masks/COVID-1.png        (COVID-Radiography)
    """
    parent = img.parent
    if parent.name.lower() != "images":
        return ""                            # masks live as a sibling of an 'images' folder
    base = parent.parent
    for mdir in ("lung masks", "masks", "lung_masks", "ManualMask"):
        cand = base / mdir / img.name
        if cand.exists():
            return str(cand.relative_to(ROOT))
    return ""


def build_manifest(registry: dict, compute_phash: bool = True):
    """Return a deduplicated, split-assigned manifest dataframe. Requires pandas.

    Steps: discover -> phash -> drop near-duplicates -> patient-level stratified split.
    """
    import pandas as pd

    df = pd.DataFrame(list(discover_rows(registry)))
    if df.empty:
        raise RuntimeError(
            "No images discovered under data/raw/. Run scripts/download_data.py first."
        )

    if compute_phash and registry["split"].get("dedup_perceptual", True):
        df["phash"] = [perceptual_hash(ROOT / p) for p in df["image_path"]]
        # drop exact phash collisions within the same source+disease (near-duplicates)
        before = len(df)
        df = df.drop_duplicates(subset=["source", "disease", "phash"], keep="first")
        print(f"dedup: removed {before - len(df)} near-duplicate(s) of {before}")

    df["split"] = _assign_splits(df, registry)
    return df.reset_index(drop=True)


def _stable_unit_fraction(key, seed):
    """Deterministic float in [0,1) for a split key, STABLE across processes/machines.

    Uses md5 (not Python's hash(), which is salted per-process via PYTHONHASHSEED) so the
    same patient/image always lands in the same split — required for reproducibility (§3.2.3).
    """
    import hashlib

    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 10_000.0


def _assign_splits(df, registry):
    """cross_source rows -> 'test'; in_domain rows -> patient-level stratified train/val/test."""
    seed = registry.get("seed", 42)
    frac = registry["split"]
    splits = []
    for _, row in df.iterrows():
        if row["role"] == "cross_source":
            splits.append("test")
            continue
        # same patient (or image, if no patient id) -> one split, deterministically
        key = row["patient_id"] if isinstance(row["patient_id"], str) else row["image_path"]
        h = _stable_unit_fraction(key, seed)
        if h < frac["train"]:
            splits.append("train")
        elif h < frac["train"] + frac["val"]:
            splits.append("val")
        else:
            splits.append("test")
    return splits


def load_registry():
    import yaml
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)
