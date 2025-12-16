"""
Script to download all pretrained model weights for future use
Downloads weights and saves to keras_pretrained_weights folder
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2,
    DenseNet121, DenseNet169, DenseNet201,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    InceptionV3, InceptionResNetV2,
    Xception, NASNetLarge, NASNetMobile
)

# Complete list of models to download
MODELS = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet50V2': ResNet50V2,
    'ResNet101V2': ResNet101V2,
    'ResNet152V2': ResNet152V2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'MobileNetV3Small': MobileNetV3Small,
    'MobileNetV3Large': MobileNetV3Large,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'EfficientNetB6': EfficientNetB6,
    'EfficientNetB7': EfficientNetB7,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'Xception': Xception,
    'NASNetLarge': NASNetLarge,
    'NASNetMobile': NASNetMobile,
}


def download_model_weights(model_name, ModelClass):
    """
    Download weights for a single model.
    
    Args:
        model_name: Name of the model
        ModelClass: Model class from keras.applications
        
    Returns:
        Success status
    """
    try:
        print(f"Downloading {model_name}...", end=' ', flush=True)
        
        # Load model with ImageNet weights (this downloads if not cached)
        model = ModelClass(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        print("✓ Done")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False


def main():
    """Download all model weights."""
    
    print("=" * 80)
    print("DOWNLOADING ALL PRETRAINED MODEL WEIGHTS")
    print("=" * 80)
    print(f"Total models to download: {len(MODELS)}")
    print(f"Weights will be cached in: ~/.keras/models/")
    print("=" * 80)
    print()
    
    success_count = 0
    failed_models = []
    
    for i, (model_name, ModelClass) in enumerate(MODELS.items(), 1):
        print(f"[{i}/{len(MODELS)}] ", end='')
        
        if download_model_weights(model_name, ModelClass):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    print()
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully downloaded: {success_count}/{len(MODELS)}")
    
    if failed_models:
        print(f"✗ Failed to download: {len(failed_models)}")
        for model in failed_models:
            print(f"  - {model}")
    
    print()
    print("Weights are cached in: ~/.keras/models/")
    print()
    print("Next steps:")
    print("1. Run: ./prepare_weights_for_drive.sh")
    print("2. Upload 'keras_pretrained_weights/' folder to Google Drive")
    print("3. Use in Colab (see COLAB_SETUP_GUIDE.md)")
    print("=" * 80)


if __name__ == '__main__':
    main()
