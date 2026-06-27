"""Robustness / SSP corroboration — P6 (PAPER_OUTLINE.md §4.8, §5.8).

Seven clinically-motivated synthetic covariate-shift perturbations (SSP). These are
*corroborating only* and explicitly NOT a contribution (§4.8) — they show the model's
accuracy degrades under whole-image shift, complementing the targeted CSA interventions.

TERMINOLOGY (§4.8): this is (b) covariate-shift perturbation — NOT (a) cross-source shift
(see cross_domain.py) and NOT (c) counterfactual intervention (see csa.py).

Each perturbation takes a float32 [0,1] image (H,W,3) + a severity and returns a perturbed
image in [0,1]. Severity grids match the legacy pipeline doc.
"""
from __future__ import annotations

import numpy as np

SEVERITIES = {
    "gaussian_noise":       [0.01, 0.03, 0.05, 0.08],
    "gaussian_blur":        [0.5, 1.0, 2.0, 3.0],
    "brightness_shift":     [-0.15, -0.10, 0.10, 0.15],
    "contrast_change":      [0.6, 0.8, 1.2, 1.5],
    "jpeg_compression":     [90, 70, 50, 30],
    "resolution_downsample": [0.75, 0.50, 0.25],
    "gamma_shift":          [0.5, 0.75, 1.5, 2.0],
}


def _clip(a):
    return np.clip(a, 0.0, 1.0).astype("float32")


def gaussian_noise(img, sigma):
    return _clip(img + np.random.normal(0, sigma, img.shape))


def gaussian_blur(img, sigma):
    from scipy.ndimage import gaussian_filter
    return _clip(gaussian_filter(img, sigma=(sigma, sigma, 0)))


def brightness_shift(img, delta):
    return _clip(img + delta)


def contrast_change(img, alpha):
    return _clip((img - 0.5) * alpha + 0.5)


def jpeg_compression(img, quality):
    import cv2
    bgr = (img[..., ::-1] * 255).astype("uint8")
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return _clip(dec[..., ::-1].astype("float32") / 255.0)


def resolution_downsample(img, ratio):
    import cv2
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(1, int(W * ratio)), max(1, int(H * ratio))),
                       interpolation=cv2.INTER_AREA)
    return _clip(cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR))


def gamma_shift(img, gamma):
    return _clip(np.power(img, gamma))


PERTURBATIONS = {
    "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur,
    "brightness_shift": brightness_shift, "contrast_change": contrast_change,
    "jpeg_compression": jpeg_compression, "resolution_downsample": resolution_downsample,
    "gamma_shift": gamma_shift,
}


def apply(img, name, severity):
    """Apply one perturbation at one severity to a single [0,1] image."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation '{name}'. Known: {list(PERTURBATIONS)}")
    return PERTURBATIONS[name](img.astype("float32"), severity)


def evaluate_robustness(model, images, y_true, perturbations=None, seed=42):
    """Per-perturbation accuracy-drop summary for one model.

    Returns {name: {severities:[...], accuracy:[...]}, clean_accuracy: float, mrr: float}.
    MRR = mean relative robustness = mean(perturbed_acc / clean_acc) across all (pert, sev).
    """
    np.random.seed(seed)
    images = np.asarray(images, dtype="float32")
    y_true = np.asarray(y_true)
    perturbations = perturbations or list(PERTURBATIONS)

    clean_pred = model.predict(images, verbose=0).argmax(1)
    clean_acc = float((clean_pred == y_true).mean())

    out, rel = {}, []
    for name in perturbations:
        accs = []
        for sev in SEVERITIES[name]:
            pert = np.stack([apply(im, name, sev) for im in images])
            pred = model.predict(pert, verbose=0).argmax(1)
            acc = float((pred == y_true).mean())
            accs.append(acc)
            rel.append(acc / (clean_acc + 1e-8))
        out[name] = {"severities": SEVERITIES[name], "accuracy": accs}
    mrr = float(np.mean(rel)) if rel else float("nan")
    return {"clean_accuracy": clean_acc, "mrr": mrr,
            "rat": float(clean_acc * mrr), "per_perturbation": out}
