"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import cv2


class Visualizer:
    """
    Visualization utilities for chest X-ray images and results.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names
        """
        if class_names is None:
            self.class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        else:
            self.class_names = class_names
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    @staticmethod
    def display_images(
        images: np.ndarray,
        labels: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        num_images: int = 16,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None
    ):
        """
        Display a grid of images.
        
        Args:
            images: Array of images
            labels: True labels
            predictions: Predicted labels
            class_names: List of class names
            num_images: Number of images to display
            figsize: Figure size
            save_path: Path to save figure
        """
        num_images = min(num_images, len(images))
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_images > 1 else [axes]
        
        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].axis('off')
            
            # Add title with label and prediction
            title = ""
            if labels is not None:
                if labels[i].ndim > 0:
                    true_label = np.argmax(labels[i])
                else:
                    true_label = int(labels[i])
                
                if class_names:
                    title += f"True: {class_names[true_label]}"
                else:
                    title += f"True: {true_label}"
            
            if predictions is not None:
                if predictions[i].ndim > 0:
                    pred_label = np.argmax(predictions[i])
                else:
                    pred_label = int(predictions[i])
                
                if class_names:
                    title += f"\nPred: {class_names[pred_label]}"
                else:
                    title += f"\nPred: {pred_label}"
                
                # Color code: green if correct, red if wrong
                if labels is not None:
                    color = 'green' if true_label == pred_label else 'red'
                    axes[i].set_title(title, color=color, fontsize=10)
                else:
                    axes[i].set_title(title, fontsize=10)
            elif title:
                axes[i].set_title(title, fontsize=10)
        
        # Hide remaining subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def compare_preprocessing(
        original: np.ndarray,
        preprocessed: np.ndarray,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Compare original and preprocessed images side by side.
        
        Args:
            original: Original image
            preprocessed: Preprocessed image
            titles: Titles for images
            save_path: Path to save figure
        """
        if titles is None:
            titles = ['Original', 'Preprocessed']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
        axes[0].set_title(titles[0], fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(preprocessed, cmap='gray' if len(preprocessed.shape) == 2 else None)
        axes[1].set_title(titles[1], fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_class_distribution(
        labels: np.ndarray,
        class_names: List[str],
        title: str = 'Class Distribution',
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of classes.
        
        Args:
            labels: Label array
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
        """
        # Convert one-hot to labels if needed
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('husl', len(class_names))
        
        bars = plt.bar(range(len(unique)), counts, color=colors, alpha=0.8, edgecolor='black')
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_augmentation(
        image: np.ndarray,
        augmenter,
        num_augmented: int = 8,
        save_path: Optional[str] = None
    ):
        """
        Visualize augmented versions of an image.
        
        Args:
            image: Input image
            augmenter: DataAugmentation instance
            num_augmented: Number of augmented images
            save_path: Path to save figure
        """
        # Get augmented images
        augmented = augmenter.augment_image(image, num_augmented=num_augmented)
        
        # Plot
        rows = int(np.ceil((num_augmented + 1) / 3))
        cols = min(3, num_augmented + 1)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_augmented > 1 else [axes]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Augmented images
        for i in range(num_augmented):
            axes[i + 1].imshow(augmented[i])
            axes[i + 1].set_title(f'Augmented {i + 1}', fontsize=12)
            axes[i + 1].axis('off')
        
        # Hide remaining subplots
        for i in range(num_augmented + 1, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Augmentation examples saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
