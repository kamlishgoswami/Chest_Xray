"""Phase 1 — two-phase transfer-learning trainer (METHODS_SECTION §3.2).

Phase 1 (feature extraction): base frozen, head trained @ lr1 = 1e-4 for E/2 epochs.
Phase 2 (fine-tuning):       base unfrozen, end-to-end @ lr2 = lr1 * 0.01 = 1e-6 for E/2 epochs.
LeNet5 (no-transfer): trains from scratch for the full E epochs at lr1.

Shared recipe: Adam, categorical cross-entropy, early stopping (patience=10, val_loss),
ReduceLROnPlateau(0.5, patience=5, min_lr=1e-8), inverse-frequency class weights,
best-val-accuracy checkpoint restored.
"""
from __future__ import annotations

LR1 = 1e-4
LR2 = LR1 * 0.01  # 1e-6
FINE_TUNE_LR_FACTOR = 0.01


def class_weights_from_labels(y_int, max_ratio=8.0):
    """Inverse-frequency class weights, CAPPED to avoid degenerate collapse (§3.2.2).

    Uncapped inverse-frequency weights can reach ~35x for very rare classes (e.g. TB),
    causing low-capacity models to collapse to the rare class. We cap the max/min weight
    ratio at `max_ratio` (default 8x) — strong enough to address imbalance, mild enough
    to keep training stable. Report the cap in the methods (it is a deliberate choice).
    """
    import numpy as np

    classes, counts = np.unique(y_int, return_counts=True)
    total = counts.sum()
    raw = {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}
    lo = min(raw.values())
    return {c: min(w, lo * max_ratio) for c, w in raw.items()}


def _callbacks(ckpt_path, monitor="val_accuracy", backup_dir=None):
    from tensorflow.keras import callbacks

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-8),
        callbacks.ModelCheckpoint(ckpt_path, monitor=monitor, save_best_only=True),
    ]
    # Per-epoch crash recovery: if backup_dir is on Drive, a Colab disconnect resumes from the
    # LAST EPOCH instead of from scratch on the next run. (No-op if backup_dir is None.)
    if backup_dir is not None:
        cbs.append(callbacks.BackupAndRestore(backup_dir=str(backup_dir)))
    return cbs


def compile_model(model, lr):
    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_two_phase(name, model, base, train_ds, val_ds, *, epochs, ckpt_path,
                    class_weight=None, backup_dir=None):
    """Run the two-phase protocol. LeNet5 (base is None) trains single-phase from scratch.

    `backup_dir` (e.g. on Google Drive) enables per-epoch crash recovery via BackupAndRestore.
    Returns the per-phase Keras History objects.
    """
    histories = {}

    if base is None:  # LeNet5 — single phase, from scratch
        compile_model(model, LR1)
        histories["scratch"] = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            class_weight=class_weight, callbacks=_callbacks(ckpt_path, backup_dir=backup_dir))
        return histories

    # Phase 1 — frozen base, train head (separate backup subdir so phases don't clash)
    base.trainable = False
    compile_model(model, LR1)
    bd1 = f"{backup_dir}/phase1" if backup_dir else None
    histories["phase1"] = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs // 2,
        class_weight=class_weight, callbacks=_callbacks(ckpt_path, backup_dir=bd1))

    # Phase 2 — unfreeze, fine-tune end-to-end at reduced lr
    base.trainable = True
    compile_model(model, LR2)
    bd2 = f"{backup_dir}/phase2" if backup_dir else None
    histories["phase2"] = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs // 2,
        class_weight=class_weight, callbacks=_callbacks(ckpt_path, backup_dir=bd2))
    return histories
