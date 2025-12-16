"""
Configuration file for the chest X-ray classification framework
"""

# Data Configuration
DATA_CONFIG = {
    'raw_data_dir': 'Dataset',
    'processed_data_dir': 'data/processed',
    'class_names': ['Normal', 'Penomonia', 'Covid', 'TB'],
    'num_classes': 4,
    'image_size': (224, 224),
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'target_size': (224, 224),
    'apply_clahe': True,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': (8, 8),
    'apply_gamma': False,
    'gamma_value': 1.2,
    'normalize': True
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20.0,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# Denoising Autoencoder Configuration
DAE_CONFIG = {
    'input_shape': (224, 224, 3),
    'latent_dim': 128,
    'noise_factor': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse'
}

# Model Configuration
MODEL_CONFIG = {
    # Available models: vgg16, vgg19, resnet50, resnet101, resnet152,
    # densenet121, densenet169, densenet201, mobilenet, mobilenetv2,
    # mobilenetv3small, mobilenetv3large, efficientnetb0-b7
    'model_name': 'resnet50',
    'input_shape': (224, 224, 3),
    'num_classes': 4,
    'weights': 'imagenet',
    'include_top': False,
    'freeze_base': True,
    'trainable_layers': None
}

# Vision Transformer Configuration
VIT_CONFIG = {
    'image_size': 224,
    'patch_size': 16,
    'num_classes': 4,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'channels': 3,
    'dropout': 0.1
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'AUC', 'Precision', 'Recall'],
    'monitor': 'val_loss',
    'patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    'use_tensorboard': True,
    'save_dir': 'checkpoints',
    'log_dir': 'logs'
}

# Fine-tuning Configuration
FINETUNING_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 1e-5,
    'unfreeze_layers': 20
}

# Class Imbalance Handling
CLASS_IMBALANCE_CONFIG = {
    'use_class_weights': True,
    'use_oversampling': False,
    'target_samples_per_class': None  # None means use max class count
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'batch_size': 32,
    'save_confusion_matrix': True,
    'save_roc_curves': True,
    'save_training_history': True,
    'results_dir': 'results'
}

# Paths
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'checkpoints': 'checkpoints',
    'logs': 'logs',
    'results': 'results',
    'dae_model': 'checkpoints/dae_model.h5'
}
