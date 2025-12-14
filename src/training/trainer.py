"""
Training pipeline for chest X-ray classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import numpy as np
from typing import Optional, List, Tuple, Dict
import os
from datetime import datetime


class Trainer:
    """
    Training pipeline with support for:
    - Transfer learning
    - Fine-tuning
    - Class imbalance handling
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: keras.Model,
        model_name: str = 'model',
        save_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            model_name: Name for saving model
            save_dir: Directory for saving checkpoints
            log_dir: Directory for logs
        """
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.history = None
    
    def setup_callbacks(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-7,
        use_tensorboard: bool = True,
        custom_callbacks: Optional[List] = None
    ) -> List:
        """
        Setup training callbacks.
        
        Args:
            monitor: Metric to monitor
            patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            reduce_lr_factor: Factor to reduce learning rate
            min_lr: Minimum learning rate
            use_tensorboard: Whether to use TensorBoard
            custom_callbacks: Additional custom callbacks
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.save_dir,
            f'{self.model_name}_best.h5'
        )
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        )
        
        # Reduce learning rate
        callbacks.append(
            ReduceLROnPlateau(
                monitor=monitor,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1
            )
        )
        
        # TensorBoard
        if use_tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_dir = os.path.join(
                self.log_dir,
                f'{self.model_name}_{timestamp}'
            )
            callbacks.append(
                TensorBoard(
                    log_dir=tensorboard_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        # CSV logger
        csv_path = os.path.join(
            self.log_dir,
            f'{self.model_name}_training.csv'
        )
        callbacks.append(
            CSVLogger(csv_path, append=True)
        )
        
        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        class_weights: Optional[Dict] = None,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            callbacks: Training callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Setup callbacks if not provided
        if callbacks is None:
            callbacks = self.setup_callbacks(
                monitor='val_loss' if X_val is not None else 'loss'
            )
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def train_with_generator(
        self,
        train_generator,
        validation_generator=None,
        epochs: int = 100,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        class_weights: Optional[Dict] = None,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model using data generators.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            class_weights: Class weights for imbalanced data
            callbacks: Training callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Setup callbacks if not provided
        if callbacks is None:
            callbacks = self.setup_callbacks(
                monitor='val_loss' if validation_generator is not None else 'loss'
            )
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        unfreeze_layers: Optional[int] = None,
        class_weights: Optional[Dict] = None,
        callbacks: Optional[List] = None
    ) -> keras.callbacks.History:
        """
        Fine-tune the model with unfrozen base layers.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Fine-tuning learning rate (typically lower)
            unfreeze_layers: Number of layers to unfreeze
            class_weights: Class weights
            callbacks: Callbacks
            
        Returns:
            Training history
        """
        # Unfreeze base model layers
        if unfreeze_layers is not None:
            for layer in self.model.layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = True
            
            # Optionally freeze some layers
            if unfreeze_layers > 0:
                for layer in self.model.layers[:-unfreeze_layers]:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        print(f"Fine-tuning with learning rate: {learning_rate}")
        print(f"Trainable layers: {sum([1 for layer in self.model.layers if layer.trainable])}")
        
        # Train
        return self.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            class_weights=class_weights,
            callbacks=callbacks
        )
    
    def get_history(self) -> Optional[keras.callbacks.History]:
        """Get training history."""
        return self.history
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the model.
        
        Args:
            path: Path to save model (if None, uses default)
        """
        if path is None:
            path = os.path.join(self.save_dir, f'{self.model_name}_final.h5')
        
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """
        Load a saved model.
        
        Args:
            path: Path to saved model
        """
        self.model = keras.models.load_model(path)
        print(f"Model loaded from: {path}")
