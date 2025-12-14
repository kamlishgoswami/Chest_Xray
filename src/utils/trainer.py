"""
Training framework for chest X-ray classification models.
Handles training, validation, and model checkpointing.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class ModelTrainer:
    """
    Unified training class for chest X-ray classification models.
    """
    
    def __init__(self, model, output_dir='outputs'):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            output_dir: Directory to save outputs (models, plots, logs)
        """
        self.model = model
        self.output_dir = output_dir
        self.history = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    def get_callbacks(self, model_name='chest_xray_model'):
        """
        Get training callbacks for model checkpointing and early stopping.
        
        Args:
            model_name: Name for saving model checkpoints
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'models', f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.output_dir, 'logs'),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train(self, 
              train_generator,
              val_generator,
              epochs=50,
              callbacks=None):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of training epochs
            callbacks: Custom callbacks (uses default if None)
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_history(self):
        """
        Save training history to JSON file.
        """
        if self.history is not None:
            history_path = os.path.join(self.output_dir, 'training_history.json')
            history_dict = {key: [float(val) for val in values] 
                          for key, values in self.history.history.items()}
            
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=4)
            
            print(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        """
        Plot training and validation metrics.
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision if available
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall if available
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'plots', 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {plot_path}")
        plt.close()
    
    def evaluate(self, test_generator, class_names=None):
        """
        Evaluate model on test data and generate metrics.
        
        Args:
            test_generator: Test data generator
            class_names: List of class names for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if class_names is None:
            class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Print and save classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        report_path = os.path.join(self.output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate and plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(self.output_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {plot_path}")
        plt.close()
