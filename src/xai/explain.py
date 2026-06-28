"""XAI corroboration of SRC — C5 (PAPER_OUTLINE.md §4.5b, P5).

Two saliency methods from DIFFERENT families (genuine cross-method robustness):
  - Grad-CAM            (CAM / gradient-based, last conv layer)
  - Integrated Gradients (path / axiomatic, pixel-level)

Quantitative faithfulness (no eyeballed heatmaps):
  - deletion AUC : remove most-salient pixels first; lower = more faithful
  - insertion AUC: add most-salient pixels first;    higher = more faithful
  - in_lung_fraction: share of |saliency| mass inside the lung mask — the INDEPENDENT
    cross-check of SRC. Hypothesis: high-SRC models put LESS saliency in the lung.

These are standard techniques; cite the specific source only after reading it (no
unverified citations in code). Images are float32 [0,1], shape (H,W,3).
"""
from __future__ import annotations

import numpy as np

BASELINE = 0.0


# ----------------------------------------------------------------- saliency maps

def _find_conv_grad_model(model):
    """Build a (inputs -> [last_conv_output, predictions]) model, descending into a nested
    backbone sub-model if needed (our zoo wraps DenseNet/ResNet/etc. as an inner Functional model).

    Returns a tf.keras.Model, or None if no 4-D conv output can be wired (e.g. ViT / pure MLP) —
    in which case the caller falls back to Integrated Gradients.
    """
    import tensorflow as tf

    def last_4d(layers):
        for layer in reversed(layers):
            try:
                shp = layer.output.shape
            except (AttributeError, TypeError):
                continue
            if shp is not None and len(shp) == 4:
                return layer
        return None

    # 1) try a conv layer directly on the top model
    layer = last_4d(model.layers)
    if layer is not None:
        try:
            return tf.keras.models.Model(model.inputs, [layer.output, model.output])
        except (ValueError, KeyError, TypeError):
            pass

    # 2) descend into a nested backbone sub-model (the common case for our transfer models)
    for sub in model.layers:
        if isinstance(sub, tf.keras.Model):
            inner = last_4d(sub.layers)
            if inner is None:
                continue
            try:
                # run the top model functionally, exposing the inner conv output as a side output
                feat_model = tf.keras.models.Model(sub.inputs, inner.output)
                inp = model.inputs
                # rebuild the forward pass so both the conv map and final preds share one graph
                x = inp
                conv_out = None
                for lyr in model.layers:
                    if lyr is sub:
                        conv_out = feat_model(x if not isinstance(x, list) else x[0])
                        x = sub(x if not isinstance(x, list) else x[0])
                    else:
                        if isinstance(lyr, tf.keras.layers.InputLayer):
                            continue
                        x = lyr(x)
                if conv_out is not None:
                    return tf.keras.models.Model(inp, [conv_out, x])
            except (ValueError, KeyError, TypeError):
                continue
    return None


def grad_cam(model, image, class_idx, layer_name=None):
    """Grad-CAM heatmap (H,W) in [0,1] for `class_idx`.

    Robust to NESTED backbones (our zoo). If the conv layer cannot be cleanly wired (ViT/MLP, or
    a graph that resists surgery), falls back to Integrated Gradients — both are in our 2-method
    suite, so the audit never crashes on a model whose internals don't expose a usable conv map.
    """
    import tensorflow as tf

    grad_model = _find_conv_grad_model(model)
    if grad_model is None:
        return integrated_gradients(model, image, class_idx)

    x = tf.convert_to_tensor(image[None].astype("float32"))
    try:
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return integrated_gradients(model, image, class_idx)
        grads = grads[0]                                     # (h,w,c)
        weights = tf.reduce_mean(grads, axis=(0, 1))         # (c,)
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # (h,w)
        cam = tf.nn.relu(cam).numpy()
        cam = _resize(cam, image.shape[:2])
        return _norm01(cam)
    except (ValueError, KeyError, TypeError, tf.errors.InvalidArgumentError):
        return integrated_gradients(model, image, class_idx)


def integrated_gradients(model, image, class_idx, steps=32):
    """Integrated Gradients attribution magnitude (H,W) in [0,1]."""
    import tensorflow as tf

    img = image.astype("float32")
    baseline = np.full_like(img, BASELINE)
    alphas = np.linspace(0, 1, steps).astype("float32")
    interp = np.stack([baseline + a * (img - baseline) for a in alphas])  # (steps,H,W,3)
    x = tf.convert_to_tensor(interp)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, x).numpy()               # (steps,H,W,3)
    avg_grads = grads.mean(axis=0)                       # (H,W,3)
    ig = (img - baseline) * avg_grads                    # (H,W,3)
    sal = np.abs(ig).sum(axis=-1)                        # (H,W)
    return _norm01(sal)


SALIENCY_METHODS = {"grad_cam": grad_cam, "integrated_gradients": integrated_gradients}


# ----------------------------------------------------------- faithfulness metrics

def _prob(model, img, class_idx):
    return float(model.predict(img[None].astype("float32"), verbose=0)[0, class_idx])


def deletion_insertion_auc(model, image, class_idx, saliency, steps=20):
    """Deletion & insertion AUC by progressively masking pixels in saliency order.

    deletion : start from full image, blank most-salient first  -> AUC LOW = faithful
    insertion: start from blank, reveal most-salient first       -> AUC HIGH = faithful
    """
    H, W = saliency.shape
    order = np.argsort(saliency.ravel())[::-1]            # most salient first
    n = H * W
    chunk = max(1, n // steps)

    blank = np.full_like(image, BASELINE)
    del_img = image.copy()
    ins_img = blank.copy()
    del_curve, ins_curve = [_prob(model, image, class_idx)], [_prob(model, blank, class_idx)]
    flat_idx = order
    for s in range(steps):
        sel = flat_idx[s * chunk:(s + 1) * chunk]
        ys, xs = np.unravel_index(sel, (H, W))
        del_img[ys, xs, :] = BASELINE
        ins_img[ys, xs, :] = image[ys, xs, :]
        del_curve.append(_prob(model, del_img, class_idx))
        ins_curve.append(_prob(model, ins_img, class_idx))
    return {"deletion_auc": float(np.trapz(del_curve) / len(del_curve)),
            "insertion_auc": float(np.trapz(ins_curve) / len(ins_curve))}


def in_lung_fraction(saliency, lung_mask):
    """Share of total |saliency| mass that falls INSIDE the lung mask (the SRC cross-check).

    mask: (H,W) or (H,W,1), 1=lung. If None, returns NaN (no mask available).
    """
    if lung_mask is None:
        return float("nan")
    m = np.asarray(lung_mask, dtype="float32")
    if m.ndim == 3:
        m = m[..., 0]
    m = _resize(m, saliency.shape)
    m = (m > 0.5).astype("float32")
    total = saliency.sum()
    return float((saliency * m).sum() / (total + 1e-8))


# ----------------------------------------------------------------- helpers

def _norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)


def _resize(a, hw):
    """Lightweight nearest/bilinear resize without extra deps (uses TF if available)."""
    if a.shape == tuple(hw):
        return a
    import tensorflow as tf
    r = tf.image.resize(a[..., None].astype("float32"), hw, method="bilinear")
    return r.numpy()[..., 0]
