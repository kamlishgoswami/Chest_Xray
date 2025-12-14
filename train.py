"""
Main training script for chest X-ray classification.
Demonstrates the unified preprocessing and explainable deep learning framework.
"""

import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.image_preprocessor import ChestXrayPreprocessor
from src.models.chest_xray_model import ChestXrayModel
from src.utils.trainer import ModelTrainer
from src.utils.explainer import ModelExplainer
from config.config import (
    MODEL_CONFIG, DATA_CONFIG, AUGMENTATION_CONFIG, 
    TRAINING_CONFIG, FINE_TUNING_CONFIG, OUTPUT_CONFIG,
    CLASS_NAMES, EXPLAINER_CONFIG
)


def main():
    """
    Main training pipeline for chest X-ray classification.
    """
    
    print("=" * 80)
    print("Chest X-ray Classification - Unified Framework")
    print("=" * 80)
    print(f"\nClasses: {', '.join(CLASS_NAMES)}")
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(f"Input Size: {MODEL_CONFIG['input_shape']}")
    print(f"Batch Size: {DATA_CONFIG['batch_size']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print("=" * 80)
    
    # Step 1: Initialize preprocessor with 224x224 resize
    print("\n[Step 1/6] Initializing preprocessor...")
    preprocessing_func = ChestXrayPreprocessor.get_preprocessing_function(
        MODEL_CONFIG['model_name']
    )
    preprocessor = ChestXrayPreprocessor(
        target_size=MODEL_CONFIG['input_shape'][:2],
        preprocessing_function=preprocessing_func
    )
    
    # Step 2: Prepare data with augmentation
    print("\n[Step 2/6] Preparing data with augmentation...")
    print(f"  - Rotation: ±{AUGMENTATION_CONFIG['rotation_range']}°")
    print(f"  - Width Shift: ±{AUGMENTATION_CONFIG['width_shift_range'] * 100}%")
    print(f"  - Height Shift: ±{AUGMENTATION_CONFIG['height_shift_range'] * 100}%")
    print(f"  - Shear: {AUGMENTATION_CONFIG['shear_range']}")
    print(f"  - Zoom: ±{AUGMENTATION_CONFIG['zoom_range'] * 100}%")
    print(f"  - Horizontal Flip: {AUGMENTATION_CONFIG['horizontal_flip']}")
    
    # Check if data directories exist
    if not os.path.exists(DATA_CONFIG['train_dir']):
        print(f"\n⚠️  Warning: Training directory not found: {DATA_CONFIG['train_dir']}")
        print("Please organize your data in the following structure:")
        print("  data/")
        print("    train/")
        print("      Normal/")
        print("      Pneumonia/")
        print("      COVID-19/")
        print("      Tuberculosis/")
        print("    validation/")
        print("      Normal/")
        print("      Pneumonia/")
        print("      COVID-19/")
        print("      Tuberculosis/")
        print("    test/")
        print("      Normal/")
        print("      Pneumonia/")
        print("      COVID-19/")
        print("      Tuberculosis/")
        print("\nSkipping training. Framework setup is complete.")
        return
    
    # Create data generators with custom augmentation parameters
    train_gen = preprocessor.create_train_generator(**AUGMENTATION_CONFIG)
    train_data = train_gen.flow_from_directory(
        DATA_CONFIG['train_dir'],
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=DATA_CONFIG['batch_size'],
        class_mode=DATA_CONFIG['class_mode'],
        shuffle=True
    )
    
    val_gen = preprocessor.create_validation_generator()
    val_data = val_gen.flow_from_directory(
        DATA_CONFIG['val_dir'],
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=DATA_CONFIG['batch_size'],
        class_mode=DATA_CONFIG['class_mode'],
        shuffle=False
    )
    
    # Step 3: Build model
    print("\n[Step 3/6] Building model...")
    model_builder = ChestXrayModel(
        model_name=MODEL_CONFIG['model_name'],
        input_shape=MODEL_CONFIG['input_shape'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    
    model = model_builder.build_cnn_model(
        trainable_layers=MODEL_CONFIG['trainable_layers']
    )
    
    model_builder.compile_model(
        learning_rate=TRAINING_CONFIG['learning_rate'],
        loss=TRAINING_CONFIG['loss']
    )
    
    print("\nModel Summary:")
    model_builder.summary()
    
    # Step 4: Train model
    print("\n[Step 4/6] Training model...")
    trainer = ModelTrainer(model, output_dir=OUTPUT_CONFIG['output_dir'])
    
    history = trainer.train(
        train_data,
        val_data,
        epochs=TRAINING_CONFIG['epochs']
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Step 5: Fine-tuning (optional)
    if FINE_TUNING_CONFIG['enabled']:
        print("\n[Step 5/6] Fine-tuning model...")
        model_builder.fine_tune(FINE_TUNING_CONFIG['unfreeze_layers'])
        
        history_ft = trainer.train(
            train_data,
            val_data,
            epochs=FINE_TUNING_CONFIG['epochs']
        )
        
        trainer.plot_training_history()
    else:
        print("\n[Step 5/6] Skipping fine-tuning (disabled in config)")
    
    # Step 6: Evaluate and generate explanations
    print("\n[Step 6/6] Evaluating model and generating explanations...")
    
    # Check if test directory exists
    if os.path.exists(DATA_CONFIG['test_dir']):
        test_gen = preprocessor.create_validation_generator()
        test_data = test_gen.flow_from_directory(
            DATA_CONFIG['test_dir'],
            target_size=MODEL_CONFIG['input_shape'][:2],
            batch_size=DATA_CONFIG['batch_size'],
            class_mode=DATA_CONFIG['class_mode'],
            shuffle=False
        )
        
        # Evaluate
        results = trainer.evaluate(test_data, class_names=CLASS_NAMES)
        
        # Generate explainability visualizations
        if EXPLAINER_CONFIG['generate_gradcam']:
            print("\nGenerating Grad-CAM visualizations...")
            explainer = ModelExplainer(model)
            
            # Get sample images from test set
            sample_images, sample_labels = next(test_data)
            num_samples = min(
                EXPLAINER_CONFIG['num_samples_to_visualize'],
                len(sample_images)
            )
            
            explainer.visualize_multiple_predictions(
                sample_images[:num_samples],
                true_labels=sample_labels[:num_samples].argmax(axis=1),
                class_names=CLASS_NAMES,
                save_path=os.path.join(OUTPUT_CONFIG['output_dir'], 'plots', 'gradcam_examples.png'),
                num_samples=num_samples
            )
    else:
        print(f"⚠️  Test directory not found: {DATA_CONFIG['test_dir']}")
        print("Skipping evaluation.")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Outputs saved to: {OUTPUT_CONFIG['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
