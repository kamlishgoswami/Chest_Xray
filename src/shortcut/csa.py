"""Counterfactual Shortcut Audit (CSA) — the CORE method, C1 (PAPER_OUTLINE.md §3.3).

Causally quantifies how much a trained classifier's predictions depend on each shortcut
channel, by intervening on that channel while preserving pathology and measuring the
change in predicted true-class probability.

Channels (§3.3.1):
    - border          : zero out a frame around the image edges (markers, text, laterality)
    - background      : replace the region OUTSIDE the lung mask with a neutral baseline
    - source_signature: normalize per-image intensity/contrast (CLAHE-like) to wash out
                        source-specific exposure signatures, keeping anatomy

Validity controls (§3.3.3) — the SOUNDNESS centerpiece:
    - sham        : identity (no-op)            -> expect ~0 effect  (negative control)
    - inside_lung : occlude a patch INSIDE lungs -> expect LARGE effect (positive control)

Causal effect (§3.3.2):  effect_c = mean_x [ f(x)[y] - f(intervene_c(x))[y] ]  with bootstrap CI.
Higher effect = the model relies more on that channel.

Images are float32 in [0,1], shape (H, W, 3). Masks are float in {0,1}, shape (H, W) or (H,W,1):
1 = lung, 0 = background.
"""
from __future__ import annotations

import numpy as np

SHORTCUT_CHANNELS = ["border", "background", "source_signature"]
CONTROL_CHANNELS = ["sham", "inside_lung"]
ALL_CHANNELS = SHORTCUT_CHANNELS + CONTROL_CHANNELS

BASELINE = 0.0  # neutral fill (black), consistent with CXR out-of-field convention


# ----------------------------------------------------------------- operators

def _as_mask(mask, hw):
    """Coerce a mask to (H, W) float in {0,1}; if None, assume an oval lung prior."""
    H, W = hw
    if mask is None:
        # coarse central-oval lung prior (used only when no segmentation is available)
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H / 2, W / 2
        m = ((yy - cy) / (0.42 * H)) ** 2 + ((xx - cx) / (0.30 * W)) ** 2 <= 1.0
        return m.astype(np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0.5).astype(np.float32)


def op_border(image, frac=0.12):
    """Zero a frame of width `frac` around the edges (border/marker channel)."""
    out = image.copy()
    H, W = image.shape[:2]
    bh, bw = int(H * frac), int(W * frac)
    mask = np.ones((H, W, 1), dtype=image.dtype)
    mask[:bh], mask[-bh:], mask[:, :bw], mask[:, -bw:] = 0, 0, 0, 0
    return out * mask + BASELINE * (1 - mask)


def op_background(image, lung_mask):
    """Replace everything OUTSIDE the lung mask with the baseline (background channel)."""
    m = _as_mask(lung_mask, image.shape[:2])[..., None]
    return image * m + BASELINE * (1 - m)


def op_source_signature(image):
    """Per-channel histogram standardization to wash out source intensity signature.

    Maps each channel to zero-mean/unit-spread then rescales to [0,1]. Preserves spatial
    anatomy (relative structure) while removing the global exposure/contrast fingerprint.
    """
    out = np.empty_like(image)
    for c in range(image.shape[-1]):
        ch = image[..., c]
        std = ch.std()
        z = (ch - ch.mean()) / (std + 1e-6)
        # squash to [0,1] via a fixed logistic so the dynamic range is source-independent
        out[..., c] = 1.0 / (1.0 + np.exp(-z))
    return out


def op_inside_lung(image, lung_mask, patch_frac=0.25):
    """Positive control: occlude a central patch INSIDE the lung region."""
    m = _as_mask(lung_mask, image.shape[:2])
    out = image.copy()
    ys, xs = np.where(m > 0.5)
    if len(ys) == 0:
        return out
    cy, cx = int(ys.mean()), int(xs.mean())
    H, W = image.shape[:2]
    ph, pw = int(H * patch_frac / 2), int(W * patch_frac / 2)
    y0, y1 = max(0, cy - ph), min(H, cy + ph)
    x0, x1 = max(0, cx - pw), min(W, cx + pw)
    out[y0:y1, x0:x1, :] = BASELINE
    return out


def intervene(image, channel, lung_mask=None):
    """Apply the counterfactual intervention for `channel`."""
    if channel == "sham":
        return image.copy()
    if channel == "border":
        return op_border(image)
    if channel == "background":
        return op_background(image, lung_mask)
    if channel == "source_signature":
        return op_source_signature(image)
    if channel == "inside_lung":
        return op_inside_lung(image, lung_mask)
    raise ValueError(f"unknown channel '{channel}'. Known: {ALL_CHANNELS}")


# --------------------------------------------------------- causal estimator

def _predict_true_prob(model, images, y_true):
    """f(x)[y_true] for a batch of images, as a 1-D array."""
    probs = model.predict(np.asarray(images), verbose=0)
    return probs[np.arange(len(y_true)), np.asarray(y_true)]


def causal_effect(model, images, y_true, channel, masks=None, n_boot=1000, seed=42):
    """Mean drop in predicted true-class prob under `channel`, with 95% bootstrap CI.

    effect = mean( f(x)[y] - f(intervene(x))[y] ).  Returns dict(effect, ci_lo, ci_hi).
    """
    images = list(images)
    masks = masks if masks is not None else [None] * len(images)
    base = _predict_true_prob(model, images, y_true)
    perturbed_imgs = [intervene(x, channel, m) for x, m in zip(images, masks)]
    pert = _predict_true_prob(model, perturbed_imgs, y_true)
    drops = base - pert  # positive => intervention reduced confidence => reliance

    rng = np.random.default_rng(seed)
    n = len(drops)
    boots = [drops[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"channel": channel, "effect": float(drops.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def run_audit(model, images, y_true, masks=None, channels=ALL_CHANNELS):
    """Full per-model audit over all channels (shortcut + controls). Returns {channel: result}."""
    return {ch: causal_effect(model, images, y_true, ch, masks) for ch in channels}


def pathology_preservation_check(model, images, y_true, masks=None, n_boot=500):
    """§4.3d — verify shortcut interventions PRESERVE in-lung pathology signal.

    A valid shortcut intervention should change predictions only via the nuisance channel, NOT by
    destroying disease evidence. We quantify this as: the inside_lung (positive control) effect
    should be LARGER than every shortcut-channel effect — i.e. removing real pathology hurts the
    prediction MORE than removing a shortcut. If a shortcut effect exceeds inside_lung, that channel
    may be co-removing pathology (a confound to flag), not just a shortcut.

    Returns per-channel {effect, exceeds_inside_lung: bool} and an overall `preserved` flag.
    """
    inside = causal_effect(model, images, y_true, "inside_lung", masks, n_boot=n_boot)["effect"]
    out = {"inside_lung_effect": float(inside), "channels": {}, "preserved": True}
    for ch in SHORTCUT_CHANNELS:
        eff = causal_effect(model, images, y_true, ch, masks, n_boot=n_boot)["effect"]
        exceeds = eff > inside
        out["channels"][ch] = {"effect": float(eff), "exceeds_inside_lung": bool(exceeds)}
        if exceeds:
            out["preserved"] = False     # this channel removes more signal than real pathology -> flag
    return out
