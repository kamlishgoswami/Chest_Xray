"""
Pretrained CNN models for transfer learning
Supports VGG, ResNet, DenseNet, MobileNet, EfficientNet, and Vision Transformers
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101, ResNet152,
    DenseNet121, DenseNet169, DenseNet201,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
)
from typing import Tuple, Optional, List


class PretrainedModels:
    """
    Factory for creating pretrained models with transfer learning from ImageNet.
    
    Supports:
    - VGG (VGG16, VGG19)
    - ResNet (ResNet50, ResNet101, ResNet152)
    - DenseNet (DenseNet121, DenseNet169, DenseNet201)
    - MobileNet (MobileNet, MobileNetV2, MobileNetV3)
    - EfficientNet (B0-B7)
    - Vision Transformers (via custom implementation)
    """
    
    AVAILABLE_MODELS = {
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'mobilenet': MobileNet,
        'mobilenetv2': MobileNetV2,
        'mobilenetv3small': MobileNetV3Small,
        'mobilenetv3large': MobileNetV3Large,
        'efficientnetb0': EfficientNetB0,
        'efficientnetb1': EfficientNetB1,
        'efficientnetb2': EfficientNetB2,
        'efficientnetb3': EfficientNetB3,
        'efficientnetb4': EfficientNetB4,
        'efficientnetb5': EfficientNetB5,
        'efficientnetb6': EfficientNetB6,
        'efficientnetb7': EfficientNetB7,
    }
    
    def __init__(
        self,
        model_name: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 4,
        weights: str = 'imagenet',
        include_top: bool = False,
        freeze_base: bool = True,
        trainable_layers: Optional[int] = None
    ):
        """
        Initialize pretrained model.
        
        Args:
            model_name: Name of the pretrained model (e.g., 'resnet50')
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes (Normal, Pneumonia, COVID-19, Tuberculosis)
            weights: Pretrained weights ('imagenet' or None)
            include_top: Whether to include top classification layer
            freeze_base: Whether to freeze base model layers
            trainable_layers: Number of layers to make trainable (from top)
        """
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.include_top = include_top
        self.freeze_base = freeze_base
        self.trainable_layers = trainable_layers
        
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model = None
        self.base_model = None
        self._build_model()
    
    def _build_model(self):
        """Build the model with transfer learning."""
        # Get the base model class
        ModelClass = self.AVAILABLE_MODELS[self.model_name]
        
        # Create base model with pretrained weights
        self.base_model = ModelClass(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )
        
        # Freeze base model if specified
        if self.freeze_base:
            self.base_model.trainable = False
        elif self.trainable_layers is not None:
            # Freeze all layers except the last n layers
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-self.trainable_layers]:
                layer.trainable = False
        
        # Build full model with custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Apply base model
        x = self.base_model(inputs, training=False)
        
        # Add custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create final model
        self.model = Model(inputs, outputs, name=f'{self.model_name}_transfer')
    
    def get_model(self) -> Model:
        """Get the compiled model."""
        return self.model
    
    def get_base_model(self) -> Model:
        """Get the base model."""
        return self.base_model
    
    def unfreeze_base_model(self, num_layers: Optional[int] = None):
        """
        Unfreeze base model for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from top (None = all)
        """
        self.base_model.trainable = True
        
        if num_layers is not None:
            # Freeze all except last num_layers
            for layer in self.base_model.layers[:-num_layers]:
                layer.trainable = False
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: Optional[List[str]] = None
    ):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', 'AUC', 'Precision', 'Recall']
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary."""
        print("=" * 80)
        print(f"MODEL: {self.model_name.upper()}")
        print("=" * 80)
        self.model.summary()
        print("\n" + "=" * 80)
        print(f"BASE MODEL: {self.model_name.upper()} (Trainable: {self.base_model.trainable})")
        print("=" * 80)
        total_params = self.base_model.count_params()
        trainable_params = sum([keras.backend.count_params(w) for w in self.base_model.trainable_weights])
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")


class VisionTransformer:
    """
    Vision Transformer (ViT) implementation for image classification.
    
    Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
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
        dropout: float = 0.1
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            image_size: Input image size
            patch_size: Size of image patches
            num_classes: Number of output classes
            dim: Dimension of embeddings
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_dim: Dimension of MLP layer
            channels: Number of input channels
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
        
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        
        self.model = None
        self._build_model()
    
    def _build_patches(self, images):
        """Extract patches from images."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def _build_model(self):
        """Build Vision Transformer model."""
        inputs = layers.Input(shape=(self.image_size, self.image_size, self.channels))
        
        # Create patches
        patches = layers.Lambda(self._build_patches)(inputs)
        
        # Patch embedding
        encoded_patches = layers.Dense(self.dim)(patches)
        
        # Add position embedding
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.dim
        )(positions)
        
        encoded_patches = encoded_patches + position_embedding
        
        # Add class token
        class_token = tf.Variable(
            initial_value=tf.random.normal([1, 1, self.dim]),
            trainable=True
        )
        class_tokens = tf.broadcast_to(
            class_token,
            [tf.shape(encoded_patches)[0], 1, self.dim]
        )
        encoded_patches = tf.concat([class_tokens, encoded_patches], axis=1)
        
        # Transformer blocks
        x = encoded_patches
        for _ in range(self.depth):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.heads,
                key_dim=self.dim // self.heads,
                dropout=self.dropout
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, x])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = layers.Dense(self.mlp_dim, activation='gelu')(x3)
            x3 = layers.Dropout(self.dropout)(x3)
            x3 = layers.Dense(self.dim)(x3)
            x3 = layers.Dropout(self.dropout)(x3)
            
            # Skip connection 2
            x = layers.Add()([x3, x2])
        
        # Layer normalization
        representation = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Extract class token
        representation = representation[:, 0]
        
        # Classification head
        representation = layers.Dropout(self.dropout)(representation)
        outputs = layers.Dense(self.num_classes, activation='softmax')(representation)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='vision_transformer')
    
    def get_model(self) -> Model:
        """Get the ViT model."""
        return self.model
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: Optional[List[str]] = None
    ):
        """Compile the model."""
        if metrics is None:
            metrics = ['accuracy', 'AUC', 'Precision', 'Recall']
        
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
