"""
Unified model framework for chest X-ray classification.
Supports both CNN and Vision Transformer (ViT) architectures.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, EfficientNetB0, EfficientNetB1
)


class ChestXrayModel:
    """
    Unified model class for chest X-ray classification.
    Supports multiple pretrained CNN architectures and ViT models.
    Classifies images into: Normal, Pneumonia, COVID-19, and Tuberculosis.
    """
    
    def __init__(self, 
                 model_name='resnet50',
                 input_shape=(224, 224, 3),
                 num_classes=4,
                 dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pretrained model to use
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (default 4: Normal, Pneumonia, COVID-19, TB)
            dropout_rate: Dropout rate for regularization
        """
        self.model_name = model_name.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        self.class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
    
    def build_cnn_model(self, trainable_layers=0):
        """
        Build a CNN-based model with transfer learning.
        
        Args:
            trainable_layers: Number of top layers to make trainable (0 means only new layers)
            
        Returns:
            Compiled Keras model
        """
        # Load base model based on model_name
        base_models = {
            'vgg16': VGG16,
            'vgg19': VGG19,
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'efficientnetb0': EfficientNetB0,
            'efficientnetb1': EfficientNetB1,
        }
        
        if self.model_name not in base_models:
            raise ValueError(f"Model {self.model_name} not supported. Choose from {list(base_models.keys())}")
        
        # Load pretrained model without top classification layer
        base_model = base_models[self.model_name](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Make top layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Build custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def build_vit_model(self):
        """
        Build a Vision Transformer (ViT) based model.
        Note: This is a placeholder for ViT integration.
        In practice, you would use models like 'vit-base-patch16-224' from transformers.
        
        Returns:
            Model architecture for ViT
        """
        # For ViT models, we typically use libraries like transformers from HuggingFace
        # This is a simplified placeholder showing the interface
        
        # Example using TensorFlow Hub or custom ViT implementation
        from tensorflow.keras.layers import Input
        
        inputs = Input(shape=self.input_shape)
        
        # In a real implementation, you would load a pretrained ViT here
        # For example, using TensorFlow Hub:
        # hub_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
        # vit_model = hub.KerasLayer(hub_url, trainable=True)
        # x = vit_model(inputs)
        
        # Placeholder: For now, we'll use a simple dense layer
        # Replace this with actual ViT implementation
        x = GlobalAveragePooling2D()(inputs)
        x = Dense(768, activation='relu')(x)  # ViT typically has 768 hidden dims
        x = Dropout(self.dropout_rate)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=predictions)
        
        return self.model
    
    def compile_model(self, 
                     learning_rate=0.001,
                     loss='categorical_crossentropy',
                     metrics=None):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function to use
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', 
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc')]
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_model(self):
        """
        Get the built model.
        
        Returns:
            Keras model
        """
        return self.model
    
    def summary(self):
        """
        Print model summary.
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build_cnn_model() or build_vit_model() first.")
    
    def fine_tune(self, num_layers):
        """
        Unfreeze and make top layers trainable for fine-tuning.
        
        Args:
            num_layers: Number of layers from the top to unfreeze
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        # Get the base model (first layer)
        base_model = self.model.layers[0] if hasattr(self.model.layers[0], 'layers') else None
        
        if base_model:
            # Unfreeze top layers
            for layer in base_model.layers[-num_layers:]:
                layer.trainable = True
            
            # Recompile with lower learning rate for fine-tuning
            self.compile_model(learning_rate=0.0001)
            print(f"Unfroze top {num_layers} layers for fine-tuning")
        else:
            print("Base model structure not found for fine-tuning")
