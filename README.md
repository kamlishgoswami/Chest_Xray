# Unified Preprocessing and Explainable Deep Learning Framework for Chest X-Ray Classification

A comprehensive deep learning framework for robust detection of **Normal**, **Pneumonia**, **COVID-19**, and **Tuberculosis** from chest X-ray images. This framework implements state-of-the-art preprocessing techniques, transfer learning with pretrained CNNs and Vision Transformers, and comprehensive evaluation metrics.

> **📁 New to this project?** Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a clear guide to all folders and files!

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Train with ResNet50 (CNN)
python3 examples/complete_pipeline.py

# Train with Vision Transformer
python3 examples/train_with_vit.py

# Compare CNN vs ViT
python3 examples/compare_cnn_vs_vit.py

# 🔥 NEW: Run ALL models (17 CNNs + ViT) with comprehensive comparison
python3 examples/comprehensive_model_comparison.py
```

📖 **See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed navigation guide**

## 🌟 Features

### Preprocessing Pipeline
- **Image Resizing**: All images resized to 224×224 for compatibility with pretrained CNN and ViT models
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast enhancement using tile-based histogram equalization with configurable clip limits, optimized for medical imaging
- **Gamma Correction**: Brightness and luminance adjustment with empirically selected gamma values
- **Denoising Autoencoders (DAEs)**: Trained with injected noise to enhance feature quality and robustness, optimized using reconstruction loss and validation loss
- **Super-Resolution Support**: Optional integration of super-resolution techniques (SRCNN, VDSR) for low-resolution X-ray images

### Data Augmentation
- **Rotations**: ±20° random rotations
- **Shifts**: 10% width/height shifts
- **Shear**: 20% shear transformation
- **Zoom**: 20% random zoom
- **Flipping**: Horizontal flipping
- Powered by `ImageDataGenerator` for improved generalization and class imbalance handling

### Model Architectures
Supports multiple pretrained models with transfer learning from ImageNet:

#### Convolutional Neural Networks (CNNs)
- **VGG**: VGG16, VGG19
- **ResNet**: ResNet50, ResNet101, ResNet152
- **DenseNet**: DenseNet121, DenseNet169, DenseNet201
- **MobileNet**: MobileNet, MobileNetV2, MobileNetV3
- **EfficientNet**: EfficientNetB0-B7

#### Vision Transformers (ViT)
- Custom Vision Transformer implementation
- Patch-based image processing
- Multi-head self-attention mechanism

### Training Features
- Transfer learning from ImageNet weights
- Fine-tuning capabilities
- Class imbalance handling (class weights, oversampling)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard integration

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves (multi-class)
- Confusion matrices
- Per-class performance metrics
- Training history visualization

## 📁 Project Structure

```
Chest_Xray/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_preprocessing.py    # Resizing, CLAHE, Gamma correction, Super-resolution
│   │   ├── denoising_autoencoder.py  # DAE implementation
│   │   └── augmentation.py           # Data augmentation and class imbalance handling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pretrained_models.py      # CNN and ViT implementations
│   │   └── model_factory.py          # Model creation utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training pipeline
│   │   └── evaluator.py              # Evaluation and metrics
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py            # Data loading utilities
│       └── visualization.py          # Visualization tools
├── configs/
│   └── config.py                     # Configuration settings
├── examples/
│   └── complete_pipeline.py          # Complete pipeline example
├── data/
│   ├── raw/                          # Raw X-ray images
│   └── processed/                    # Preprocessed images
├── checkpoints/                      # Saved models
├── logs/                             # Training logs
├── results/                          # Evaluation results
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/kamlishgoswami/Chest_Xray.git
cd Chest_Xray
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start

Run the complete pipeline example:
```bash
python examples/complete_pipeline.py
```

### Custom Pipeline

```python
import numpy as np
from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.models import PretrainedModels
from src.training import Trainer, Evaluator
from src.utils import DataLoader

# 1. Load and preprocess data
data_loader = DataLoader(
    data_dir='data/raw',
    class_names=['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
)

preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    apply_clahe=True,
    clahe_clip_limit=2.0,
    apply_gamma=False
)

# 2. Setup augmentation
augmenter = DataAugmentation(
    rotation_range=20.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 3. Create model
model_builder = PretrainedModels(
    model_name='resnet50',
    num_classes=4,
    weights='imagenet',
    freeze_base=True
)
model = model_builder.get_model()

# 4. Compile and train
model_builder.compile_model(
    optimizer='adam',
    learning_rate=0.001
)

trainer = Trainer(model=model, model_name='resnet50')
history = trainer.train(X_train, y_train, X_val, y_val, epochs=100)

# 5. Evaluate
evaluator = Evaluator(model=model)
results = evaluator.evaluate(X_test, y_test)
evaluator.print_evaluation_results(results)
```

### Using Denoising Autoencoder

```python
from src.preprocessing import DenoisingAutoencoder

# Create and train DAE
dae = DenoisingAutoencoder(
    input_shape=(224, 224, 3),
    latent_dim=128,
    noise_factor=0.2
)

dae.compile_model(optimizer='adam', learning_rate=0.001)
dae.train(X_train, X_val, epochs=50, batch_size=32)

# Denoise images
denoised_images = dae.denoise(noisy_images)
```

### Using Vision Transformer

```python
from src.models import VisionTransformer

# Create ViT model
vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=4,
    dim=768,
    depth=12,
    heads=12
)

model = vit.get_model()
vit.compile_model(optimizer='adam', learning_rate=0.001)
```

## 📊 Data Format

The framework expects data organized in the following structure:

```
data/raw/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Pneumonia/
│   ├── image1.jpg
│   └── ...
├── COVID-19/
│   ├── image1.jpg
│   └── ...
└── Tuberculosis/
    ├── image1.jpg
    └── ...
```

## ⚙️ Configuration

All configuration settings are centralized in `configs/config.py`:

- **Data Configuration**: Image sizes, class names, split ratios
- **Preprocessing Configuration**: CLAHE settings, gamma correction
- **Augmentation Configuration**: Rotation, shift, shear, zoom parameters
- **Model Configuration**: Model selection, transfer learning settings
- **Training Configuration**: Epochs, batch size, learning rate, callbacks
- **Evaluation Configuration**: Metrics, visualization settings

## 🔬 Technical Details

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Tile-based histogram equalization**: Divides image into tiles and applies histogram equalization
- **Clip limit**: Prevents over-amplification of noise (default: 2.0)
- **Tile size**: Grid size for tiles (default: 8×8)
- **Medical imaging optimized**: Particularly effective for X-ray images

### Transfer Learning
- **ImageNet pretrained weights**: Leverages knowledge from large-scale image classification
- **Freezable base layers**: Option to freeze pretrained layers during initial training
- **Fine-tuning support**: Unfreeze layers for domain-specific optimization
- **Custom classification head**: Dense layers with dropout for overfitting prevention

### Class Imbalance Handling
- **Class weights**: Automatically computed based on class distribution
- **Oversampling**: Augmentation-based oversampling of minority classes
- **Stratified splitting**: Maintains class distribution across train/val/test sets

## 📈 Results and Visualization

The framework automatically generates:
- Confusion matrices (normalized and raw)
- ROC curves for each class
- Training history plots (loss, accuracy, learning rate)
- Per-class performance metrics
- Class distribution visualizations

## � Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project organization guide
- **[docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md)** - Implementation details & algorithms
- **[docs/VIT_GUIDE.md](docs/VIT_GUIDE.md)** - Vision Transformer usage guide- **[docs/COMPREHENSIVE_COMPARISON_GUIDE.md](docs/COMPREHENSIVE_COMPARISON_GUIDE.md)** - 🔥 NEW: Run all models comparison- **[docs/COLAB_SETUP_GUIDE.md](docs/COLAB_SETUP_GUIDE.md)** - Google Colab setup instructions
- **[docs/UNUSED_FEATURES.md](docs/UNUSED_FEATURES.md)** - Optional features guide

## 🔗 References

### Super-Resolution Techniques
- **SRCNN**: Image Super-Resolution Using Deep Convolutional Networks (IEEE Conference)
- **VDSR**: Accurate Image Super-Resolution Using Very Deep Convolutional Networks (IEEE Conference)

### Vision Transformers
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021

---

**🎯 Ready to start?** Run `python3 examples/complete_pipeline.py` after activating your environment!

### Transfer Learning
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Huang et al., "Densely Connected Convolutional Networks", CVPR 2017
- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019

## 📝 License

This project is available for academic and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This framework is designed for research and educational purposes. For clinical applications, proper validation and regulatory approval are required.