#!/bin/bash
# Script to package Keras model weights for Google Drive

echo "=========================================="
echo "Packaging Keras Model Weights for Google Drive"
echo "=========================================="

# Create a directory to store weights
WEIGHTS_DIR="keras_pretrained_weights"
mkdir -p "$WEIGHTS_DIR"

# Copy all downloaded weights
echo "Copying weights from ~/.keras/models/..."
cp -v ~/.keras/models/*.h5 "$WEIGHTS_DIR/" 2>/dev/null || echo "No .h5 files found yet"
cp -v ~/.keras/models/*.pb "$WEIGHTS_DIR/" 2>/dev/null || true
cp -v ~/.keras/models/*.index "$WEIGHTS_DIR/" 2>/dev/null || true

# Create a README
cat > "$WEIGHTS_DIR/README.txt" << 'EOF'
Keras Pretrained Model Weights
================================

These are pretrained ImageNet weights for various architectures.

TO USE IN GOOGLE COLAB:
------------------------
1. Upload this folder to your Google Drive
2. In your Colab notebook, add:

   from google.colab import drive
   drive.mount('/content/drive')
   
   import os
   os.environ['KERAS_HOME'] = '/content/drive/MyDrive/keras_pretrained_weights'

3. Now when you load models, they'll use these cached weights!

MODELS INCLUDED:
----------------
- VGG16, VGG19
- ResNet50, ResNet101, ResNet152
- DenseNet121, DenseNet169, DenseNet201
- MobileNet, MobileNetV2
- EfficientNetB0-B7

Each model's weights are in the .h5 files.
EOF

# Check total size
TOTAL_SIZE=$(du -sh "$WEIGHTS_DIR" | cut -f1)
echo ""
echo "=========================================="
echo "✓ Weights packaged successfully!"
echo "=========================================="
echo "Location: $WEIGHTS_DIR/"
echo "Total Size: $TOTAL_SIZE"
echo ""
echo "NEXT STEPS:"
echo "1. Upload the '$WEIGHTS_DIR' folder to Google Drive"
echo "2. In Colab, set: os.environ['KERAS_HOME'] = '/content/drive/MyDrive/$WEIGHTS_DIR'"
echo "3. Models will load from Drive instead of downloading!"
echo "=========================================="
