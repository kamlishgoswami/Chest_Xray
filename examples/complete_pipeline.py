"""
Example script demonstrating the complete pipeline for chest X-ray classification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.preprocessing import ImagePreprocessor, DenoisingAutoencoder, DataAugmentation
from src.preprocessing.augmentation import ClassImbalanceHandler
from src.models import ModelFactory
from src.training import Trainer, Evaluator
from src.utils import DataLoader, Visualizer
from configs.config import *


def main():
    """
    Complete pipeline for chest X-ray classification.
    """
    
    print("=" * 80)
    print("UNIFIED PREPROCESSING AND DEEP LEARNING FRAMEWORK")
    print("Chest X-Ray Classification: Normal, Pneumonia, COVID-19, Tuberculosis")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    data_loader = DataLoader(
        data_dir=DATA_CONFIG['raw_data_dir'],
        class_names=DATA_CONFIG['class_names']
    )
    
    # Example: Load images from directory
    # Uncomment when you have data available
    # images, labels = data_loader.load_images_from_directory(
    #     DATA_CONFIG['raw_data_dir'],
    #     target_size=DATA_CONFIG['image_size']
    # )
    
    # For demonstration, create dummy data
    print("Note: Using dummy data for demonstration. Replace with actual data loading.")
    num_samples = 400
    images = np.random.rand(num_samples, 224, 224, 3)
    labels = np.random.randint(0, 4, size=num_samples)
    
    # Visualize class distribution
    visualizer = Visualizer(class_names=DATA_CONFIG['class_names'])
    visualizer.plot_class_distribution(
        labels,
        DATA_CONFIG['class_names'],
        save_path='results/class_distribution.png'
    )
    
    # ========================================================================
    # Step 2: Preprocess Images
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING")
    print("=" * 80)
    
    preprocessor = ImagePreprocessor(**PREPROCESSING_CONFIG)
    
    print("Applying preprocessing pipeline...")
    print(f"  - Resizing to {PREPROCESSING_CONFIG['target_size']}")
    print(f"  - CLAHE: {PREPROCESSING_CONFIG['apply_clahe']}")
    print(f"  - Gamma correction: {PREPROCESSING_CONFIG['apply_gamma']}")
    
    # Apply preprocessing
    # preprocessed_images = preprocessor.preprocess_batch(images)
    preprocessed_images = images  # Already preprocessed for dummy data
    
    # ========================================================================
    # Step 3: Train Denoising Autoencoder (Optional)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: DENOISING AUTOENCODER (Optional)")
    print("=" * 80)
    
    # Uncomment to train DAE
    # dae = DenoisingAutoencoder(**DAE_CONFIG)
    # dae.compile_model(
    #     optimizer=DAE_CONFIG['optimizer'],
    #     learning_rate=DAE_CONFIG['learning_rate'],
    #     loss=DAE_CONFIG['loss']
    # )
    # dae.get_model_summary()
    # 
    # Split data for DAE training
    # X_train_dae, X_val_dae = train_test_split(preprocessed_images, test_size=0.2)
    # 
    # Train DAE
    # dae_history = dae.train(
    #     X_train_dae, X_val_dae,
    #     epochs=DAE_CONFIG['epochs'],
    #     batch_size=DAE_CONFIG['batch_size'],
    #     save_path=PATHS['dae_model']
    # )
    # 
    # Apply denoising
    # preprocessed_images = dae.denoise(preprocessed_images)
    
    print("Skipping DAE training for demonstration.")
    
    # ========================================================================
    # Step 4: Data Augmentation and Split
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DATA AUGMENTATION AND SPLITTING")
    print("=" * 80)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data(
        preprocessed_images,
        labels,
        test_size=DATA_CONFIG['test_size'],
        val_size=DATA_CONFIG['val_size'],
        random_state=DATA_CONFIG['random_state']
    )
    
    # Setup augmentation
    augmenter = DataAugmentation(**AUGMENTATION_CONFIG)
    
    # Handle class imbalance
    if CLASS_IMBALANCE_CONFIG['use_class_weights']:
        class_weights = ClassImbalanceHandler.compute_class_weights(y_train)
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None
    
    # ========================================================================
    # Step 5: Build Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: BUILDING MODEL")
    print("=" * 80)
    
    print(f"Creating {MODEL_CONFIG['model_name'].upper()} model with transfer learning...")
    
    # Create model
    from src.models import PretrainedModels
    
    model_builder = PretrainedModels(**MODEL_CONFIG)
    model = model_builder.get_model()
    
    # Compile model
    model_builder.compile_model(
        optimizer=TRAINING_CONFIG['optimizer'],
        learning_rate=TRAINING_CONFIG['learning_rate'],
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics']
    )
    
    model_builder.summary()
    
    # ========================================================================
    # Step 6: Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING")
    print("=" * 80)
    
    trainer = Trainer(
        model=model,
        model_name=MODEL_CONFIG['model_name'],
        save_dir=TRAINING_CONFIG['save_dir'],
        log_dir=TRAINING_CONFIG['log_dir']
    )
    
    # Train
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        class_weights=class_weights
    )
    
    # ========================================================================
    # Step 7: Fine-tuning (Optional)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: FINE-TUNING (Optional)")
    print("=" * 80)
    
    # Uncomment to perform fine-tuning
    # model_builder.unfreeze_base_model(num_layers=FINETUNING_CONFIG['unfreeze_layers'])
    # 
    # history_finetune = trainer.fine_tune(
    #     X_train, y_train,
    #     X_val, y_val,
    #     epochs=FINETUNING_CONFIG['epochs'],
    #     batch_size=FINETUNING_CONFIG['batch_size'],
    #     learning_rate=FINETUNING_CONFIG['learning_rate'],
    #     class_weights=class_weights
    # )
    
    print("Skipping fine-tuning for demonstration.")
    
    # ========================================================================
    # Step 8: Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: EVALUATION")
    print("=" * 80)
    
    evaluator = Evaluator(
        model=model,
        class_names=DATA_CONFIG['class_names']
    )
    
    # Evaluate on test set
    results = evaluator.evaluate(
        X_test, y_test,
        batch_size=EVALUATION_CONFIG['batch_size']
    )
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    # Save visualizations
    os.makedirs(EVALUATION_CONFIG['results_dir'], exist_ok=True)
    
    if EVALUATION_CONFIG['save_confusion_matrix']:
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=os.path.join(EVALUATION_CONFIG['results_dir'], 'confusion_matrix.png')
        )
    
    if EVALUATION_CONFIG['save_roc_curves']:
        y_pred_probs = model.predict(X_test)
        evaluator.plot_roc_curves(
            y_test, y_pred_probs,
            save_path=os.path.join(EVALUATION_CONFIG['results_dir'], 'roc_curves.png')
        )
    
    if EVALUATION_CONFIG['save_training_history']:
        evaluator.plot_training_history(
            history,
            save_path=os.path.join(EVALUATION_CONFIG['results_dir'], 'training_history.png')
        )
    
    # Save final model
    trainer.save_model()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model saved to: {TRAINING_CONFIG['save_dir']}")
    print(f"Results saved to: {EVALUATION_CONFIG['results_dir']}")


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    main()
