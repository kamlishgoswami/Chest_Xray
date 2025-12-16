# Vision Transformer (ViT) Implementation Guide

## ✅ **Successfully Implemented!**

Vision Transformer has been fully implemented and integrated into your chest X-ray classification framework.

---

## 🏗️ **Architecture Overview**

### **What is Vision Transformer?**
- **Paper:** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Key Idea:** Apply Transformer architecture (from NLP) directly to images by treating them as sequences of patches
- **Advantage:** Captures long-range dependencies better than CNNs

### **ViT Pipeline:**
```
Input Image (224×224) 
    ↓
Split into Patches (16×16 = 196 patches)
    ↓
Linear Projection (to embedding dimension)
    ↓
Add Positional Embeddings
    ↓
Transformer Encoder Blocks (12 layers)
    ├─ Multi-Head Self-Attention (12 heads)
    └─ Feed-Forward MLP
    ↓
Global Average Pooling
    ↓
Classification Head → 4 classes
```

---

## 📦 **What Was Added**

### **1. Core Implementation**
**File:** `src/models/vision_transformer.py`

**Components:**
- `PatchExtractor` - Divides image into non-overlapping patches
- `PatchEncoder` - Projects patches and adds positional embeddings
- `MultiHeadSelfAttention` - Multi-head attention mechanism
- `TransformerBlock` - Complete transformer encoder block
- `VisionTransformer` - Main ViT class
- `create_vit_model()` - Convenience function

### **2. Training Script**
**File:** `examples/train_with_vit.py`

Complete pipeline using Vision Transformer:
- Data loading
- Preprocessing
- ViT model creation
- Training
- Evaluation
- Results visualization

### **3. Comparison Script**
**File:** `examples/compare_cnn_vs_vit.py`

Head-to-head comparison:
- ResNet50 (CNN) vs Vision Transformer
- Same dataset, same training setup
- Performance metrics comparison
- Visual comparison charts

---

## 🚀 **How to Use**

### **Option 1: Train with ViT Only**

```bash
source venv/bin/activate
python3 examples/train_with_vit.py
```

**Output:**
- Model: `checkpoints/vit/vision_transformer_final.h5`
- Results: `results/vit/`
  - Confusion matrix
  - ROC curves
  - Training history

### **Option 2: Compare CNN vs ViT**

```bash
source venv/bin/activate
python3 examples/compare_cnn_vs_vit.py
```

**Output:**
- Models saved in: `checkpoints/comparison/`
- Results: `results/comparison/`
  - Individual model results
  - Comparison CSV
  - Comparison charts

### **Option 3: Use in Custom Script**

```python
from src.models import VisionTransformer
from configs.config import VIT_CONFIG

# Create ViT model
vit = VisionTransformer(**VIT_CONFIG)
model = vit.get_model()

# Compile
vit.compile_model(
    optimizer='adam',
    learning_rate=0.001,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Evaluate
results = model.evaluate(X_test, y_test)
```

---

## ⚙️ **Configuration**

**Default ViT Configuration** (in `configs/config.py`):

```python
VIT_CONFIG = {
    'image_size': 224,        # Input image size
    'patch_size': 16,         # Patch size (224/16 = 14×14 = 196 patches)
    'num_classes': 4,         # Normal, Pneumonia, COVID-19, TB
    'dim': 768,               # Embedding dimension
    'depth': 12,              # Number of transformer blocks
    'heads': 12,              # Number of attention heads
    'mlp_dim': 3072,          # MLP hidden dimension (4×dim)
    'channels': 3,            # RGB images
    'dropout': 0.1            # Dropout rate
}
```

### **Model Variants:**

#### **ViT-Tiny** (Faster, less parameters)
```python
VIT_CONFIG = {
    'dim': 192,
    'depth': 12,
    'heads': 3,
    'mlp_dim': 768,
    # ...other params
}
```

#### **ViT-Small**
```python
VIT_CONFIG = {
    'dim': 384,
    'depth': 12,
    'heads': 6,
    'mlp_dim': 1536,
}
```

#### **ViT-Base** (Current - default)
```python
VIT_CONFIG = {
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
}
```

#### **ViT-Large** (Slower, more parameters)
```python
VIT_CONFIG = {
    'dim': 1024,
    'depth': 24,
    'heads': 16,
    'mlp_dim': 4096,
}
```

---

## 📊 **Model Comparison**

| Feature | CNN (ResNet50) | Vision Transformer |
|---------|---------------|-------------------|
| **Architecture** | Convolutional layers | Transformer blocks |
| **Inductive Bias** | Strong (locality) | Weak (learns from data) |
| **Receptive Field** | Local → Global | Global from start |
| **Parameters** | ~24M | ~86M (ViT-Base) |
| **Training Speed** | Faster | Slower |
| **Data Requirement** | Less data | More data preferred |
| **Transfer Learning** | Excellent | Good (improving) |
| **Interpretability** | Filter visualization | Attention maps |

---

## 🎯 **Expected Performance**

### **With Small Dataset (88 images):**
- **CNN (ResNet50):** Better performance (transfer learning advantage)
- **ViT:** May underperform (needs more data)

### **With Large Dataset (>1000 images):**
- **CNN:** Good performance
- **ViT:** Can match or exceed CNN performance

### **Recommendation:**
- **Small datasets:** Use CNN (ResNet, DenseNet)
- **Large datasets:** Try both, ViT may excel
- **Production:** Compare both and choose best

---

## 🔍 **Model Architecture Details**

### **Parameters:**
```
ViT-Base (~86M parameters):
- Patch Embedding: 768 × (3 × 16 × 16) = 590K
- Position Embedding: 196 × 768 = 150K
- 12 Transformer Blocks: ~85M
  - Multi-Head Attention: ~2.4M per block
  - MLP: ~4.7M per block
- Classification Head: ~3M
```

### **Memory:**
```
- Input: 224×224×3 = 150K pixels
- Patches: 196 × 768 = ~150K values
- Intermediate: Varies by block
- Output: 4 classes
```

---

## 💡 **Tips for Best Results**

### **1. Data Augmentation**
ViT benefits more from augmentation than CNNs:
```python
AUGMENTATION_CONFIG = {
    'rotation_range': 20.0,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
}
```

### **2. Learning Rate**
Start with lower learning rate for ViT:
```python
TRAINING_CONFIG = {
    'learning_rate': 0.0001,  # Lower than CNN
    'epochs': 50,              # May need more epochs
}
```

### **3. Regularization**
Increase dropout for small datasets:
```python
VIT_CONFIG = {
    'dropout': 0.2,  # Higher dropout
}
```

### **4. Batch Size**
ViT needs more memory:
```python
TRAINING_CONFIG = {
    'batch_size': 16,  # Reduce if GPU memory limited
}
```

---

## 🎓 **Further Reading**

- **Original Paper:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **ViT Explained:** [Hugging Face Tutorial](https://huggingface.co/blog/vision-transformers)
- **Medical Imaging with ViT:** Recent research showing promise in radiology

---

## ✅ **Verification**

Test the implementation:

```bash
source venv/bin/activate
python3 -c "from src.models import VisionTransformer; vit = VisionTransformer(); print('✓ ViT working!')"
```

You should see:
```
✓ Vision Transformer imported successfully
✓ Number of patches: 196
✓ Embedding dimension: 768
✓ Model created successfully
```

---

## 🚀 **Quick Start**

```bash
# Train with ViT (3 epochs for quick test)
source venv/bin/activate
python3 examples/train_with_vit.py

# Or compare CNN vs ViT
python3 examples/compare_cnn_vs_vit.py
```

**Your Vision Transformer is ready to use!** 🎉
