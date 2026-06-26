"""Phase 1 — the 7-model zoo (PAPER_OUTLINE.md §6, METHODS_SECTION.md §3.2.1, §4.2).

One representative architecture per family + a no-transfer-learning baseline:

    DenseNet201        base-paper champion
    EfficientNetB0     modern efficient CNN
    ResNet50           canonical residual baseline
    MobileNetV3Large   edge/deployment representative
    Xception           depthwise-separable family
    ViT                transformer (documented lower-bound recipe)
    LeNet5             no-transfer baseline (trained from scratch)

All pretrained backbones share ONE classification head for fair comparison:
    GAP -> Dense(512, relu) -> Dropout(0.5) -> Dense(256, relu) -> Dropout(0.5) -> Dense(C, softmax)

ViT is handled separately (patch-based; via tf.keras or HuggingFace). LeNet5 bypasses
transfer learning entirely (§3.2.1) and is the isolate-pretraining baseline.
"""
from __future__ import annotations

IMG_SIZE = (224, 224)
NUM_CLASSES = 4

# name -> (keras.applications builder attribute, preprocess module path)
BACKBONES = {
    "densenet201":      ("DenseNet201",      "densenet"),
    "efficientnetb0":   ("EfficientNetB0",   "efficientnet"),
    "resnet50":         ("ResNet50",         "resnet50"),
    "mobilenetv3large": ("MobileNetV3Large", "mobilenet_v3"),
    "xception":         ("Xception",         "xception"),
}
CUSTOM_MODELS = {"lenet5", "vit"}
MODEL_NAMES = list(BACKBONES) + ["vit", "lenet5"]


def _make_vit_layers():
    """Lazily define ViT custom layers (needs TF at call time, not import time)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    class ClassToken(layers.Layer):
        """Learnable CLS token, broadcast to the batch."""

        def build(self, input_shape):
            self.cls = self.add_weight(
                shape=(1, 1, input_shape[-1]), initializer="zeros", trainable=True, name="cls")

        def call(self, x):
            b = tf.shape(x)[0]
            return tf.broadcast_to(self.cls, (b, 1, tf.shape(x)[-1]))

    class AddPositionalEmbedding(layers.Layer):
        """Learnable positional embedding added to the token sequence."""

        def __init__(self, n_tokens, dim, **kw):
            super().__init__(**kw)
            self.n_tokens, self.dim = n_tokens, dim

        def build(self, _):
            self.pos = self.add_weight(
                shape=(1, self.n_tokens, self.dim), initializer="random_normal",
                trainable=True, name="pos")

        def call(self, x):
            return x + self.pos

    return ClassToken, AddPositionalEmbedding


def _classification_head(x, num_classes, name="head"):
    """Shared head across all pretrained backbones (METHODS_SECTION §3.2.1)."""
    from tensorflow.keras import layers

    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dense(512, activation="relu", name=f"{name}_fc1")(x)
    x = layers.Dropout(0.5, name=f"{name}_do1")(x)
    x = layers.Dense(256, activation="relu", name=f"{name}_fc2")(x)
    x = layers.Dropout(0.5, name=f"{name}_do2")(x)
    return layers.Dense(num_classes, activation="softmax", name=f"{name}_out")(x)


def build_backbone_model(name, num_classes=NUM_CLASSES, img_size=IMG_SIZE, freeze_base=True):
    """Build a pretrained-backbone model with the shared head. Returns (model, base).

    Each backbone's REQUIRED preprocess_input is baked in as a layer, so the model accepts
    raw [0,1] images (uniform loader) and applies its own ImageNet preprocessing internally.
    Critical: feeding raw [0,1] to e.g. MobileNetV3/ResNet (which expect [-1,1] / caffe mean
    subtraction) cripples the pretrained features -> chance-level accuracy.
    """
    import importlib

    import tensorflow as tf
    from tensorflow.keras import Model, applications

    builder_attr, preprocess_mod = BACKBONES[name]
    builder = getattr(applications, builder_attr)
    preprocess_input = importlib.import_module(
        f"tensorflow.keras.applications.{preprocess_mod}").preprocess_input

    base = builder(include_top=False, weights="imagenet", input_shape=(*img_size, 3))
    base.trainable = not freeze_base  # Phase 1 freezes; Phase 2 unfreezes

    inputs = tf.keras.Input(shape=(*img_size, 3))           # raw [0,1]
    x = tf.keras.layers.Rescaling(255.0)(inputs)            # back to [0,255]
    x = preprocess_input(x)                                 # backbone-specific normalization
    x = base(x, training=False)
    outputs = _classification_head(x, num_classes, name=name)
    return Model(inputs, outputs, name=name), base


def build_lenet5(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Modified LeNet-5, from scratch — the no-transfer baseline (§4.2)."""
    from tensorflow.keras import Model, Input, layers

    inputs = Input(shape=(*img_size, 3))
    x = layers.Conv2D(6, 5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(16, 5, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(120, 5, activation="relu")(x)
    outputs = _classification_head(x, num_classes, name="lenet5")
    return Model(inputs, outputs, name="lenet5")


def build_vit(num_classes=NUM_CLASSES, img_size=IMG_SIZE, patch_size=16,
              proj_dim=192, depth=6, heads=6, mlp_dim=384):
    """Compact native Keras-3 Vision Transformer (no external deps).

    Self-contained (patch embed -> [+CLS, +pos] -> transformer blocks -> head) so it runs
    identically on Mac and Colab without transformers/keras_cv version conflicts. This is a
    from-scratch ViT (no ImageNet pretrain); per METHODS_SECTION §4.2 ViT is documented as a
    performance lower-bound under the shared recipe. Accepts raw [0,1] images.
    """
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    ClassToken, AddPositionalEmbedding = _make_vit_layers()
    H, W = img_size
    n_patches = (H // patch_size) * (W // patch_size)

    inputs = tf.keras.Input(shape=(*img_size, 3))
    # patch embedding via a strided conv
    x = layers.Conv2D(proj_dim, patch_size, strides=patch_size, name="vit_patch")(inputs)
    x = layers.Reshape((n_patches, proj_dim))(x)
    # prepend CLS token
    cls = ClassToken(name="vit_cls")(x)
    x = layers.Concatenate(axis=1)([cls, x])
    # positional embedding
    x = AddPositionalEmbedding(n_patches + 1, proj_dim, name="vit_pos")(x)

    for i in range(depth):
        # MHSA block
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.MultiHeadAttention(num_heads=heads, key_dim=proj_dim // heads,
                                      dropout=0.1, name=f"vit_mha_{i}")(y, y)
        x = layers.Add()([x, y])
        # MLP block
        y = layers.LayerNormalization(epsilon=1e-6)(x)
        y = layers.Dense(mlp_dim, activation="gelu")(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(proj_dim)(y)
        x = layers.Add()([x, y])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_out = x[:, 0]                                   # CLS token representation
    # shared-style head (Dense path; no GAP since we already pooled via CLS)
    h = layers.Dense(512, activation="relu", name="vit_fc1")(cls_out)
    h = layers.Dropout(0.5)(h)
    h = layers.Dense(256, activation="relu", name="vit_fc2")(h)
    h = layers.Dropout(0.5)(h)
    outputs = layers.Dense(num_classes, activation="softmax", name="vit_out")(h)
    return Model(inputs, outputs, name="vit")


def build_model(name, num_classes=NUM_CLASSES, img_size=IMG_SIZE, freeze_base=True):
    """Factory: dispatch by name. Returns (model, base_or_None)."""
    name = name.lower()
    if name in BACKBONES:
        return build_backbone_model(name, num_classes, img_size, freeze_base)
    if name == "lenet5":
        return build_lenet5(num_classes, img_size), None
    if name == "vit":
        return build_vit(num_classes, img_size), None
    raise ValueError(f"unknown model '{name}'. Known: {MODEL_NAMES}")
