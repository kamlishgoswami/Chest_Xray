# Chest X-ray Classification Framework

A unified preprocessing and explainable deep learning framework for robust detection of **Normal**, **Pneumonia**, **COVID-19**, and **Tuberculosis** from chest X-ray images.

## Overview

This framework provides a comprehensive solution for chest X-ray image classification with the following key features:

- **Unified Preprocessing**: All images are resized to 224×224 to ensure compatibility with pretrained CNN and ViT models
- **Data Augmentation**: Advanced augmentation techniques (rotation, shift, shear, zoom, flipping) to improve generalization and mitigate class imbalance
- **Transfer Learning**: Support for multiple pretrained models (VGG16, VGG19, ResNet50, ResNet101, EfficientNet)
- **Explainable AI**: Grad-CAM visualizations for interpretable model predictions
- **Multi-class Classification**: Detects Normal, Pneumonia, COVID-19, and Tuberculosis

## Features

### 1. Unified Preprocessing Pipeline
- Automatic resizing to 224×224 pixels
- Model-specific preprocessing functions
- Normalization and standardization
- Compatible with CNN and Vision Transformer architectures

### 2. Data Augmentation
The framework implements comprehensive data augmentation to improve model robustness:
- **Rotation**: ±20° random rotation
- **Width Shift**: ±20% horizontal translation
- **Height Shift**: ±20% vertical translation
- **Shear**: 0.2 shear intensity
- **Zoom**: ±20% random zoom
- **Horizontal Flip**: Random horizontal flipping

### 3. Model Architecture
- Support for multiple pretrained models:
  - VGG16, VGG19
  - ResNet50, ResNet101
  - EfficientNetB0, EfficientNetB1
- Custom classification head with dropout for regularization
- Fine-tuning capabilities for improved performance

### 4. Explainability
- Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
- Heatmap overlays on original images
- Confidence scores for all classes

## Installation

### Requirements
- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kamlishgoswami/Chest_Xray.git
cd Chest_Xray
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Chest_Xray/
├── config/
│   └── config.py                 # Configuration file
├── src/
│   ├── preprocessing/
│   │   └── image_preprocessor.py # Preprocessing and augmentation
│   ├── models/
│   │   └── chest_xray_model.py   # Model architectures
│   └── utils/
│       ├── trainer.py            # Training utilities
│       └── explainer.py          # Explainability tools
├── data/                         # Data directory (not included)
│   ├── train/
│   ├── validation/
│   └── test/
├── train.py                      # Main training script
├── predict.py                    # Inference script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data Organization

Organize your data in the following structure:

```
data/
├── train/
│   ├── Normal/
│   ├── Pneumonia/
│   ├── COVID-19/
│   └── Tuberculosis/
├── validation/
│   ├── Normal/
│   ├── Pneumonia/
│   ├── COVID-19/
│   └── Tuberculosis/
└── test/
    ├── Normal/
    ├── Pneumonia/
    ├── COVID-19/
    └── Tuberculosis/
```

## Usage

### Training

1. Configure the model and training parameters in `config/config.py`

2. Run the training script:
```bash
python train.py
```

The training process will:
- Load and preprocess data with augmentation
- Build the specified model architecture
- Train with early stopping and learning rate reduction
- Save the best model based on validation accuracy
- Generate training history plots
- Evaluate on test set with metrics and confusion matrix
- Create Grad-CAM visualizations

### Inference

To make predictions on new images:

```bash
python predict.py --image-path path/to/xray.jpg
```

Optional arguments:
- `--model-path`: Path to trained model (default: `outputs/models/chest_xray_classifier_best.h5`)
- `--model-type`: Model type for preprocessing (default: `resnet50`)
- `--no-explanation`: Disable Grad-CAM visualization

Example:
```bash
python predict.py --image-path test_image.jpg --model-path outputs/models/chest_xray_classifier_best.h5
```

### Configuration

Edit `config/config.py` to customize:

#### Model Configuration
```python
MODEL_CONFIG = {
    'model_name': 'resnet50',      # Choose pretrained model
    'input_shape': (224, 224, 3),  # Fixed for compatibility
    'num_classes': 4,               # 4 disease classes
    'dropout_rate': 0.5,           # Regularization
}
```

#### Augmentation Configuration
```python
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
}
```

#### Training Configuration
```python
TRAINING_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
}
```

## Outputs

All outputs are saved in the `outputs/` directory:

- `models/`: Saved model checkpoints
- `plots/`: Training history, confusion matrix, Grad-CAM visualizations
- `logs/`: TensorBoard logs
- `training_history.json`: Training metrics
- `classification_report.json`: Detailed classification metrics

## Technical Details

### Preprocessing Pipeline

1. **Image Loading**: Load images from directory structure
2. **Resizing**: All images resized to 224×224 pixels
3. **Augmentation** (training only):
   - Random rotation (±20°)
   - Random shifts (±20%)
   - Shear transformations
   - Zoom variations
   - Horizontal flips
4. **Normalization**: Model-specific preprocessing (ImageNet normalization for pretrained models)

### Model Architecture

The framework uses transfer learning with pretrained models:

1. **Base Model**: Pretrained on ImageNet (frozen initially)
2. **Global Average Pooling**: Reduce spatial dimensions
3. **Dense Layers**: 512 → 256 neurons with ReLU activation
4. **Dropout**: Regularization (default 0.5)
5. **Output Layer**: 4 neurons with softmax activation

### Training Process

1. **Initial Training**: Train with frozen base model
2. **Callbacks**:
   - ModelCheckpoint: Save best model
   - EarlyStopping: Prevent overfitting
   - ReduceLROnPlateau: Adaptive learning rate
   - TensorBoard: Training visualization
3. **Fine-tuning** (optional): Unfreeze top layers for additional training

### Explainability

The framework implements Grad-CAM for visual explanations:

1. Extract feature maps from last convolutional layer
2. Compute gradients of predicted class with respect to feature maps
3. Weight feature maps by gradients
4. Generate heatmap showing important regions
5. Overlay heatmap on original image

## Performance Metrics

The framework tracks comprehensive metrics:
- Accuracy
- Precision (per class and weighted average)
- Recall (per class and weighted average)
- F1-Score
- AUC-ROC
- Confusion Matrix

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{chest_xray_framework,
  title={Unified Preprocessing and Explainable Deep Learning Framework for Chest X-ray Classification},
  author={Chest X-ray Research Team},
  year={2024},
  url={https://github.com/kamlishgoswami/Chest_Xray}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Pretrained models from TensorFlow/Keras Applications
- Grad-CAM implementation based on the original paper by Selvaraju et al.

## Contact

For questions or issues, please open an issue on GitHub.