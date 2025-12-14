"""
Explainability module for chest X-ray classification.
Provides visualization of model decisions using Grad-CAM and other techniques.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2


class ModelExplainer:
    """
    Class for generating explainable AI visualizations for chest X-ray models.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the layer to visualize (defaults to last conv layer)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
    
    def _find_last_conv_layer(self):
        """
        Automatically find the last convolutional layer in the model.
        
        Returns:
            Name of the last convolutional layer
        """
        for layer in reversed(self.model.layers):
            # Check if layer is Conv2D
            if 'conv' in layer.name.lower():
                return layer.name
        
        # If no conv layer found in main model, check base model
        if hasattr(self.model.layers[0], 'layers'):
            for layer in reversed(self.model.layers[0].layers):
                if 'conv' in layer.name.lower():
                    return layer.name
        
        return None
    
    def generate_gradcam(self, image, class_index=None):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (preprocessed, shape: (1, 224, 224, 3))
            class_index: Index of class to visualize (None for predicted class)
            
        Returns:
            Heatmap array
        """
        if self.layer_name is None:
            raise ValueError("No convolutional layer found for Grad-CAM")
        
        # Create a model that outputs both predictions and conv layer output
        grad_model = Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layer_name).output, 
                    self.model.output]
        )
        
        # Compute gradient of predicted class with respect to feature map
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_gradcam(self, image, heatmap, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image (224, 224, 3)
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor for overlay
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize original image if needed
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Overlay heatmap on image
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def visualize_prediction(self, image, true_label=None, class_names=None, save_path=None):
        """
        Create comprehensive visualization with prediction and Grad-CAM.
        
        Args:
            image: Input image (preprocessed for model)
            true_label: True class label (optional)
            class_names: List of class names
            save_path: Path to save visualization (optional)
        """
        if class_names is None:
            class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        
        # Get prediction
        predictions = self.model.predict(image, verbose=0)
        pred_class = np.argmax(predictions[0])
        pred_confidence = predictions[0][pred_class]
        
        # Generate Grad-CAM
        heatmap = self.generate_gradcam(image, pred_class)
        
        # Prepare image for visualization (denormalize if needed)
        vis_image = image[0].copy()
        if vis_image.max() <= 1.0:
            vis_image = np.uint8(255 * vis_image)
        
        # Overlay Grad-CAM
        overlayed = self.overlay_gradcam(vis_image, heatmap)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(vis_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlayed image
        axes[2].imshow(overlayed)
        title = f'Prediction: {class_names[pred_class]}\nConfidence: {pred_confidence:.2%}'
        if true_label is not None:
            title += f'\nTrue Label: {class_names[true_label]}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return {
            'predicted_class': class_names[pred_class],
            'confidence': float(pred_confidence),
            'all_predictions': {class_names[i]: float(predictions[0][i]) 
                              for i in range(len(class_names))}
        }
    
    def visualize_multiple_predictions(self, images, true_labels=None, 
                                      class_names=None, save_path=None, num_samples=4):
        """
        Visualize predictions for multiple images in a grid.
        
        Args:
            images: Batch of images
            true_labels: True labels (optional)
            class_names: List of class names
            save_path: Path to save visualization
            num_samples: Number of samples to visualize
        """
        if class_names is None:
            class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_samples):
            image = images[idx:idx+1]
            
            # Get prediction
            predictions = self.model.predict(image, verbose=0)
            pred_class = np.argmax(predictions[0])
            pred_confidence = predictions[0][pred_class]
            
            # Generate Grad-CAM
            heatmap = self.generate_gradcam(image, pred_class)
            
            # Prepare image for visualization
            vis_image = image[0].copy()
            if vis_image.max() <= 1.0:
                vis_image = np.uint8(255 * vis_image)
            
            # Overlay Grad-CAM
            overlayed = self.overlay_gradcam(vis_image, heatmap)
            
            # Plot
            axes[idx, 0].imshow(vis_image)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(heatmap, cmap='jet')
            axes[idx, 1].set_title('Grad-CAM Heatmap')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(overlayed)
            title = f'Pred: {class_names[pred_class]} ({pred_confidence:.2%})'
            if true_labels is not None:
                title += f'\nTrue: {class_names[true_labels[idx]]}'
            axes[idx, 2].set_title(title)
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
