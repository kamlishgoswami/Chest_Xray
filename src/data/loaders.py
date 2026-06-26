"""Phase 1 — manifest -> tf.data pipeline (METHODS_SECTION §3.1.2, §3.2.4).

Reads data/manifest.csv (built by scripts/build_manifest.py) and yields batched,
resized, normalized tensors. Supports filtering by split and by source (for the
cross-source experiments in Phase 3).

Clinically-constrained augmentation (§3.2.4) is applied to the TRAIN split only.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["Covid", "Normal", "Pneumonia", "TB"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_manifest():
    import pandas as pd

    path = ROOT / "data" / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError("data/manifest.csv missing. Run scripts/build_manifest.py first.")
    return pd.read_csv(path)


def filter_rows(df, split=None, sources=None, roles=None):
    """Subset the manifest by split / source(s) / role(s)."""
    if split is not None:
        df = df[df["split"] == split]
    if sources is not None:
        df = df[df["source"].isin(sources)]
    if roles is not None:
        df = df[df["role"].isin(roles)]
    return df


def _decode(path, label, img_size, training, augment):
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    if training and augment:
        # clinically-constrained (§3.2.4): mild rotation/shift via flip+brightness only here;
        # heavier geometric aug handled by a Keras augmentation layer in the training script.
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(df, *, img_size=(224, 224), batch_size=8, training=False,
                 augment=True, shuffle=True, seed=42):
    """Build a tf.data.Dataset of (image, one_hot_label) from a manifest subset."""
    import tensorflow as tf

    paths = [str(ROOT / p) for p in df["image_path"]]
    labels = [CLASS_TO_IDX[c] for c in df["disease"]]
    one_hot = tf.one_hot(labels, depth=len(CLASSES))

    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))
    if shuffle and training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2048), seed=seed)
    ds = ds.map(lambda p, y: _decode(p, y, img_size, training, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def labels_int(df):
    """Integer labels for class-weight computation."""
    return [CLASS_TO_IDX[c] for c in df["disease"]]
