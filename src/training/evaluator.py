"""
Model evaluation and metrics
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Model evaluation with comprehensive metrics.
    """
    
    def __init__(
        self,
        model: keras.Model,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            class_names: List of class names
        """
        self.model = model
        
        if class_names is None:
            self.class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        else:
            self.class_names = class_names
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            batch_size: Batch size for evaluation
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred_probs = self.model.predict(X_test, batch_size=batch_size, verbose=verbose)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Convert one-hot to labels if needed
        if y_test.ndim > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        # Compute metrics
        results = {}
        
        # Model evaluation
        eval_results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        metric_names = self.model.metrics_names
        for name, value in zip(metric_names, eval_results):
            results[name] = value
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        results['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # ROC AUC (multi-class)
        if y_test.ndim > 1:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_probs, average='weighted', multi_class='ovr')
                results['roc_auc_weighted'] = roc_auc
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not compute ROC AUC: {e}")
        
        # Per-class metrics
        results['per_class_metrics'] = {}
        for i, class_name in enumerate(self.class_names):
            if class_name in report:
                results['per_class_metrics'][class_name] = report[class_name]
        
        return results
    
    def print_evaluation_results(self, results: Dict):
        """
        Print evaluation results.
        
        Args:
            results: Results from evaluate()
        """
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        # Overall metrics
        print("\nOverall Metrics:")
        for key, value in results.items():
            if key not in ['classification_report', 'confusion_matrix', 'per_class_metrics']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        
        for class_name in self.class_names:
            if class_name in results['per_class_metrics']:
                metrics = results['per_class_metrics'][class_name]
                print(
                    f"{class_name:<15} "
                    f"{metrics['precision']:<12.4f} "
                    f"{metrics['recall']:<12.4f} "
                    f"{metrics['f1-score']:<12.4f} "
                    f"{int(metrics['support']):<10}"
                )
        
        # Weighted average
        if 'weighted avg' in results['classification_report']:
            metrics = results['classification_report']['weighted avg']
            print("-" * 80)
            print(
                f"{'Weighted Avg':<15} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1-score']:<12.4f} "
                f"{int(metrics['support']):<10}"
            )
        
        print("=" * 80)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Path to save figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_roc_curves(
        self,
        y_test: np.ndarray,
        y_pred_probs: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot ROC curves for each class.
        
        Args:
            y_test: True labels (one-hot encoded)
            y_pred_probs: Predicted probabilities
            save_path: Path to save figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Compute ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.close()
    
    def plot_training_history(
        self,
        history: keras.callbacks.History,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Plot training history.
        
        Args:
            history: Training history
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history.history:
            axes[2].plot(history.history['lr'])
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].set_yscale('log')
            axes[2].grid(alpha=0.3)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {save_path}")
        
        plt.close()
