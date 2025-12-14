# Technical Implementation Details

## Overview

This document provides detailed technical information about the unified preprocessing and deep learning framework for chest X-ray classification.

## Preprocessing Pipeline

### 1. Image Resizing (224×224)

All images are resized to 224×224 pixels to ensure compatibility with pretrained CNN and Vision Transformer models. This standardization is crucial for:
- **Transfer Learning**: Pretrained models from ImageNet expect this input size
- **Batch Processing**: Uniform sizes enable efficient batch processing
- **Memory Efficiency**: Consistent dimensions optimize GPU memory usage

**Implementation**: Uses OpenCV's cubic interpolation (`cv2.INTER_CUBIC`) for high-quality resizing.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE enhances local contrast through tile-based histogram equalization, particularly effective for medical imaging.

**Parameters**:
- `clip_limit`: 2.0 (prevents over-amplification of noise)
- `tile_grid_size`: 8×8 (divides image into 64 tiles)

**Algorithm**:
1. Divide image into tiles (8×8 grid)
2. Apply histogram equalization to each tile
3. Clip histogram to prevent noise amplification
4. Interpolate between tiles for smooth transitions

**For RGB Images**: Applied to L-channel in LAB color space to preserve color information.

### 3. Gamma Correction

Adjusts brightness and luminance using power-law transformation:

```
output = (input / 255)^(1/gamma) * 255
```

**Gamma Values**:
- γ > 1: Brightens the image
- γ < 1: Darkens the image
- γ = 1.2: Default (empirically selected for X-rays)

### 4. Denoising Autoencoders (DAE)

**Architecture**:
```
Encoder:
  Input (224, 224, 3)
  → Conv2D(64) → MaxPool
  → Conv2D(128) → MaxPool
  → Conv2D(256) → MaxPool
  → Conv2D(512)
  → Flatten → Dense(latent_dim=128)

Decoder:
  Dense(latent_dim) → Reshape
  → Conv2D(512) → UpSample
  → Conv2D(256) → UpSample
  → Conv2D(128) → UpSample
  → Conv2D(64)
  → Conv2D(3, activation='sigmoid')
```

**Training Process**:
1. Add Gaussian noise: `noisy = image + noise_factor * N(0,1)`
2. Train to reconstruct original from noisy input
3. Optimize MSE loss between output and original
4. Monitor validation loss for early stopping

**Benefits**:
- Removes image artifacts
- Enhances feature quality
- Improves model robustness
- Reduces overfitting

### 5. Super-Resolution (Optional)

**Techniques**:
- **SRCNN**: 3-layer CNN for single image super-resolution
- **VDSR**: Very deep (20 layers) super-resolution network
- **Bicubic**: Traditional interpolation baseline

**Use Cases**:
- Low-resolution X-ray images (<224×224)
- Upscaling before preprocessing
- Quality enhancement

## Data Augmentation

### ImageDataGenerator Parameters

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| Rotation | ±20° | Handle slight positioning variations |
| Width Shift | 10% | Horizontal translation invariance |
| Height Shift | 10% | Vertical translation invariance |
| Shear | 20% | Handle perspective distortions |
| Zoom | 20% | Scale invariance |
| Horizontal Flip | Yes | Mirror symmetry |
| Vertical Flip | No | Anatomically incorrect |

### Benefits

1. **Increased Training Data**: Generates infinite variations
2. **Generalization**: Reduces overfitting on training set
3. **Class Imbalance**: Oversamples minority classes
4. **Robustness**: Handles real-world variations

## Model Architectures

### Supported CNN Models

#### VGG (VGG16, VGG19)
- **Depth**: 16-19 layers
- **Architecture**: Stacked 3×3 convolutions
- **Parameters**: 138M (VGG16)
- **Best for**: High accuracy, less sensitive to augmentation

#### ResNet (ResNet50, ResNet101, ResNet152)
- **Depth**: 50-152 layers
- **Innovation**: Skip connections (residual learning)
- **Parameters**: 25M-60M
- **Best for**: Very deep networks, gradient flow

#### DenseNet (DenseNet121, DenseNet169, DenseNet201)
- **Depth**: 121-201 layers
- **Innovation**: Dense connections between all layers
- **Parameters**: 8M-20M
- **Best for**: Feature reuse, parameter efficiency

#### MobileNet (V1, V2, V3)
- **Depth**: Variable
- **Innovation**: Depthwise separable convolutions
- **Parameters**: 3M-5M
- **Best for**: Mobile/edge deployment, speed

#### EfficientNet (B0-B7)
- **Depth**: Scales with version
- **Innovation**: Compound scaling (depth, width, resolution)
- **Parameters**: 5M-66M
- **Best for**: Best accuracy/efficiency tradeoff

### Vision Transformer (ViT)

**Architecture**:
```
1. Patch Embedding:
   - Divide 224×224 image into 16×16 patches
   - Total patches: 196 (14×14)
   - Embed each patch to dim=768

2. Positional Encoding:
   - Add learnable position embeddings
   - Preserve spatial information

3. Transformer Encoder (×12):
   - Multi-Head Self-Attention (12 heads)
   - Layer Normalization
   - MLP (3072 hidden units)
   - Residual connections

4. Classification Head:
   - Extract [CLS] token
   - Dense layer → 4 classes
```

**Parameters**:
- Patch size: 16×16
- Embedding dim: 768
- Depth: 12 transformer blocks
- Attention heads: 12
- MLP dim: 3072
- Total parameters: ~86M

## Transfer Learning

### Strategy

1. **Phase 1: Feature Extraction**
   - Freeze pretrained base model
   - Train only classification head
   - Learning rate: 1e-3
   - Epochs: 50-100

2. **Phase 2: Fine-tuning** (Optional)
   - Unfreeze top N layers
   - Very low learning rate: 1e-5
   - Epochs: 20-50
   - Prevents catastrophic forgetting

### Custom Classification Head

```python
GlobalAveragePooling2D()
→ BatchNormalization()
→ Dropout(0.5)
→ Dense(512, activation='relu')
→ BatchNormalization()
→ Dropout(0.3)
→ Dense(256, activation='relu')
→ Dropout(0.2)
→ Dense(4, activation='softmax')
```

**Design Choices**:
- **GlobalAveragePooling**: Reduces spatial dimensions, prevents overfitting
- **BatchNormalization**: Stabilizes training, faster convergence
- **Dropout Layers**: Progressive rates (0.5 → 0.3 → 0.2)
- **Dense Layers**: 512 → 256 → 4 (gradual reduction)

## Training Pipeline

### Loss Functions

**Categorical Cross-Entropy**:
```
L = -Σ(y_true * log(y_pred))
```

Best for multi-class classification with one-hot encoding.

### Optimizers

1. **Adam** (Default)
   - Adaptive learning rate
   - Momentum + RMSprop
   - Learning rate: 1e-3

2. **SGD with Momentum**
   - Classic approach
   - Momentum: 0.9
   - Learning rate: 1e-2

### Learning Rate Scheduling

**ReduceLROnPlateau**:
- Monitor: validation loss
- Patience: 5 epochs
- Factor: 0.5 (halve learning rate)
- Min LR: 1e-7

### Callbacks

1. **EarlyStopping**: Stops training when validation loss plateaus (patience=10)
2. **ModelCheckpoint**: Saves best model based on validation loss
3. **TensorBoard**: Logs metrics for visualization
4. **CSVLogger**: Saves training history to CSV

## Class Imbalance Handling

### Method 1: Class Weights

Compute balanced weights:
```python
weight[i] = n_samples / (n_classes * n_samples_class[i])
```

### Method 2: Oversampling

1. Identify minority classes
2. Generate augmented samples using DataAugmentation
3. Balance dataset to target_samples per class

### Stratified Splitting

Maintains class distribution across train/val/test splits:
- Training: 70%
- Validation: 10%
- Test: 20%

## Evaluation Metrics

### Classification Metrics

1. **Accuracy**: Overall correctness
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under receiver operating characteristic curve

### Multi-Class Handling

- **One-vs-Rest (OvR)**: Binary classification for each class
- **Weighted Average**: Account for class imbalance
- **Per-Class Metrics**: Individual performance per disease

### Visualization

1. **Confusion Matrix**: Shows prediction patterns
2. **ROC Curves**: Per-class discriminative ability
3. **Training History**: Loss and accuracy over time

## Computational Requirements

### Memory Requirements

| Model | Parameters | GPU Memory | Training Time |
|-------|-----------|------------|---------------|
| MobileNetV2 | 3.5M | 2-4 GB | ~30 min/epoch |
| ResNet50 | 25M | 4-6 GB | ~45 min/epoch |
| EfficientNetB0 | 5M | 3-5 GB | ~35 min/epoch |
| VGG16 | 138M | 8-10 GB | ~60 min/epoch |
| ViT-Base | 86M | 6-8 GB | ~50 min/epoch |

*Based on batch_size=32, 1000 samples/epoch, NVIDIA V100*

### Recommended Hardware

**Minimum**:
- GPU: 6 GB VRAM
- RAM: 16 GB
- Storage: 50 GB

**Recommended**:
- GPU: 16 GB VRAM (RTX 4000, V100)
- RAM: 32 GB
- Storage: 100 GB SSD

## References

### Key Papers

1. **CLAHE**: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
2. **ResNet**: He et al. (2016). "Deep Residual Learning for Image Recognition"
3. **DenseNet**: Huang et al. (2017). "Densely Connected Convolutional Networks"
4. **EfficientNet**: Tan & Le (2019). "EfficientNet: Rethinking Model Scaling"
5. **ViT**: Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words"
6. **SRCNN**: Dong et al. (2014). "Image Super-Resolution Using Deep CNNs"
7. **VDSR**: Kim et al. (2016). "Accurate Image Super-Resolution Using Very Deep CNNs"
8. **Denoising Autoencoders**: Vincent et al. (2008). "Extracting and Composing Robust Features"

### Medical Imaging Applications

1. Kermany et al. (2018). "Identifying Medical Diagnoses from Chest X-Rays"
2. Wang et al. (2020). "COVID-Net: Deep Neural Network for COVID-19 Detection"
3. Rajpurkar et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection"

## Best Practices

### For Medical Imaging

1. **Always preprocess**: CLAHE improves contrast for X-rays
2. **Use transfer learning**: Limited medical data benefits from ImageNet
3. **Handle class imbalance**: Medical datasets often imbalanced
4. **Validate rigorously**: Use stratified k-fold cross-validation
5. **Document thoroughly**: Medical applications require transparency
6. **Ensemble models**: Combine predictions from multiple models
7. **Calibrate probabilities**: Medical decisions require confidence scores

### Hyperparameter Tuning

1. Start with pretrained model + frozen base
2. Try learning rates: [1e-2, 1e-3, 1e-4]
3. Experiment with augmentation intensity
4. Fine-tune only if sufficient data (>1000 samples/class)
5. Use early stopping to prevent overfitting
6. Monitor validation metrics closely

## Troubleshooting

### Common Issues

**Problem**: Model overfitting (high train accuracy, low val accuracy)
**Solutions**:
- Increase dropout rates
- Add more augmentation
- Reduce model complexity
- Get more training data

**Problem**: Model underfitting (low train and val accuracy)
**Solutions**:
- Increase model capacity
- Train for more epochs
- Reduce regularization
- Check data quality

**Problem**: Class imbalance affecting performance
**Solutions**:
- Use class weights
- Oversample minority classes
- Use focal loss
- Stratified sampling

**Problem**: Out of memory errors
**Solutions**:
- Reduce batch size
- Use smaller model (MobileNet, EfficientNetB0)
- Use mixed precision training
- Reduce image resolution
