"""
Configuration file for chest X-ray classification.
Contains all hyperparameters and settings.
"""

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'resnet50',  # Options: vgg16, vgg19, resnet50, resnet101, efficientnetb0, efficientnetb1
    'input_shape': (224, 224, 3),  # Fixed size for compatibility with pretrained models
    'num_classes': 4,  # Normal, Pneumonia, COVID-19, Tuberculosis
    'dropout_rate': 0.5,
    'trainable_layers': 0,  # Number of base model layers to fine-tune (0 = freeze all)
}

# Data Configuration
DATA_CONFIG = {
    'train_dir': 'data/train',
    'val_dir': 'data/validation',
    'test_dir': 'data/test',
    'batch_size': 32,
    'class_mode': 'categorical',
}

# Preprocessing and Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,  # Degrees
    'width_shift_range': 0.2,  # Fraction of total width
    'height_shift_range': 0.2,  # Fraction of total height
    'shear_range': 0.2,  # Shear intensity
    'zoom_range': 0.2,  # Range for random zoom
    'horizontal_flip': True,  # Random horizontal flip
    'fill_mode': 'nearest',  # Fill mode for new pixels
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'loss': 'categorical_crossentropy',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
}

# Fine-tuning Configuration
FINE_TUNING_CONFIG = {
    'enabled': False,
    'unfreeze_layers': 20,  # Number of layers to unfreeze from top
    'learning_rate': 0.0001,  # Lower learning rate for fine-tuning
    'epochs': 20,
}

# Output Configuration
OUTPUT_CONFIG = {
    'output_dir': 'outputs',
    'model_save_name': 'chest_xray_classifier',
}

# Class Names
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']

# Explainability Configuration
EXPLAINER_CONFIG = {
    'generate_gradcam': True,
    'num_samples_to_visualize': 4,
}
