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
