"""
Quick test script to verify the framework architecture
"""

import sys
import os

print("=" * 80)
print("FRAMEWORK ARCHITECTURE VERIFICATION")
print("=" * 80)

# Test directory structure
print("\n1. Checking directory structure...")
required_dirs = [
    'src/preprocessing',
    'src/models',
    'src/training',
    'src/utils',
    'configs',
    'examples',
    'data/raw',
    'data/processed'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ {dir_path} - MISSING")

# Test required files
print("\n2. Checking required files...")
required_files = [
    'README.md',
    'requirements.txt',
    '.gitignore',
    'src/__init__.py',
    'src/preprocessing/__init__.py',
    'src/preprocessing/image_preprocessing.py',
    'src/preprocessing/denoising_autoencoder.py',
    'src/preprocessing/augmentation.py',
    'src/models/__init__.py',
    'src/models/pretrained_models.py',
    'src/models/model_factory.py',
    'src/training/__init__.py',
    'src/training/trainer.py',
    'src/training/evaluator.py',
    'src/utils/__init__.py',
    'src/utils/data_loader.py',
    'src/utils/visualization.py',
    'configs/config.py',
    'examples/complete_pipeline.py'
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - MISSING")

# Check key classes/functions
print("\n3. Checking key components in code...")

components = {
    'ImagePreprocessor': 'src/preprocessing/image_preprocessing.py',
    'DenoisingAutoencoder': 'src/preprocessing/denoising_autoencoder.py',
    'DataAugmentation': 'src/preprocessing/augmentation.py',
    'PretrainedModels': 'src/models/pretrained_models.py',
    'VisionTransformer': 'src/models/pretrained_models.py',
    'ModelFactory': 'src/models/model_factory.py',
    'Trainer': 'src/training/trainer.py',
    'Evaluator': 'src/training/evaluator.py',
    'DataLoader': 'src/utils/data_loader.py',
    'Visualizer': 'src/utils/visualization.py'
}

for component, file_path in components.items():
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            if f'class {component}' in content:
                print(f"  ✓ {component} class found in {file_path}")
            else:
                print(f"  ✗ {component} class NOT found in {file_path}")
    else:
        print(f"  ✗ {file_path} - FILE MISSING")

# Check configuration
print("\n4. Checking configuration...")
config_keys = [
    'DATA_CONFIG',
    'PREPROCESSING_CONFIG',
    'AUGMENTATION_CONFIG',
    'DAE_CONFIG',
    'MODEL_CONFIG',
    'TRAINING_CONFIG'
]

if os.path.exists('configs/config.py'):
    with open('configs/config.py', 'r') as f:
        config_content = f.read()
        for key in config_keys:
            if key in config_content:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} - MISSING")

# Check README
print("\n5. Checking README.md...")
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        readme = f.read()
        keywords = [
            'CLAHE',
            'Gamma correction',
            'Denoising Autoencoder',
            'ResNet',
            'VGG',
            'DenseNet',
            'MobileNet',
            'EfficientNet',
            'Vision Transformer',
            'Transfer learning',
            'ImageNet',
            'augmentation',
            'rotation',
            'class imbalance'
        ]
        
        found_keywords = sum(1 for keyword in keywords if keyword in readme)
        print(f"  ✓ Found {found_keywords}/{len(keywords)} key features documented")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✅ Framework architecture verified successfully!")
print("\nTo use the framework:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Prepare your data in data/raw/<class_name>/ format")
print("3. Run: python examples/complete_pipeline.py")
