"""
Vision Transformer (ViT) implementation for chest X-ray classification
Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class PatchExtractor(layers.Layer):
    """
    Extract patches from input images.
    Divides image into non-overlapping patches.
    """
    
    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """
    Encode patches with linear projection and add positional embeddings.
    """
    
    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout_layer = layers.Dropout(dropout)
    
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout_layer(weights)
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        # Apply attention
        attention, weights = self.attention(query, key, value)
        
        # Combine heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Transformer encoder block with multi-head attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        # Layer norm 1 + Multi-head attention + Residual
        x1 = self.layernorm1(inputs)
        attention_output = self.att(x1, training=training)
        attention_output = self.dropout1(attention_output, training=training)
        x2 = layers.Add()([attention_output, inputs])
        
        # Layer norm 2 + MLP + Residual
        x3 = self.layernorm2(x2)
        x3 = self.mlp(x3, training=training)
        x3 = self.dropout2(x3, training=training)
        output = layers.Add()([x3, x2])
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate
        })
        return config


class VisionTransformer:
    """
    Vision Transformer for image classification.
    
    Architecture:
    1. Split image into patches
    2. Linear projection of patches
    3. Add positional embeddings
    4. Pass through transformer encoder blocks
    5. Classification head
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 4,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        channels: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            image_size: Input image size (assumes square images)
            patch_size: Size of image patches
            num_classes: Number of output classes
            dim: Embedding dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_dim: Hidden dimension of MLP
            channels: Number of image channels
            dropout: Dropout rate
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dropout = dropout
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the Vision Transformer model."""
        
        inputs = layers.Input(shape=(self.image_size, self.image_size, self.channels))
        
        # Create patches
        patches = PatchExtractor(self.patch_size)(inputs)
        
        # Encode patches
        encoded_patches = PatchEncoder(self.num_patches, self.dim)(patches)
        
        # Dropout
        x = layers.Dropout(self.dropout)(encoded_patches)
        
        # Transformer blocks
        for _ in range(self.depth):
            x = TransformerBlock(
                embed_dim=self.dim,
                num_heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout
            )(x)
        
        # Layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dropout
        x = layers.Dropout(self.dropout)(x)
        
        # Classification head
        x = layers.Dense(self.mlp_dim, activation=tf.nn.gelu)(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='vision_transformer')
    
    def get_model(self):
        """Return the Keras model."""
        return self.model
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def get_config(self):
        """Get model configuration."""
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'mlp_dim': self.mlp_dim,
            'channels': self.channels,
            'dropout': self.dropout,
            'num_patches': self.num_patches,
            'patch_dim': self.patch_dim
        }


# Convenience function to create ViT model
def create_vit_model(
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 4,
    dim: int = 768,
    depth: int = 12,
    heads: int = 12,
    mlp_dim: int = 3072,
    channels: int = 3,
    dropout: float = 0.1
) -> keras.Model:
    """
    Create a Vision Transformer model.
    
    Args:
        image_size: Input image size
        patch_size: Size of patches
        num_classes: Number of output classes
        dim: Embedding dimension
        depth: Number of transformer blocks
        heads: Number of attention heads
        mlp_dim: MLP hidden dimension
        channels: Number of input channels
        dropout: Dropout rate
        
    Returns:
        Keras Model
    """
    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dropout=dropout
    )
    return vit.get_model()
