"""
Comprehensive Model Comparison Script
Runs ALL CNN architectures and Vision Transformer, compares results,
and provides detailed logging from input image to output results.

This script:
1. Loads and preprocesses the dataset with detailed logging
2. Trains all available CNN models (VGG, ResNet, DenseNet, MobileNet, EfficientNet)
3. Trains Vision Transformer
4. Collects comprehensive metrics for all models
5. Creates comparison visualizations and reports
6. Logs every step from input to final predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import time

from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.preprocessing.augmentation import ClassImbalanceHandler
from src.models import PretrainedModels, VisionTransformer
from src.training import Trainer, Evaluator
from src.utils import DataLoader, Visualizer
from configs.config import *

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# All available CNN models
CNN_MODELS = [
    'vgg16', 'vgg19',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet169', 'densenet201',
    'mobilenet', 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large',
    'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4'
]


class ComprehensiveLogger:
    """Detailed logging for the entire pipeline"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logs = {
            'dataset': {},
            'preprocessing': {},
            'models': {},
            'training': {},
            'evaluation': {},
            'comparison': {}
        }
        
    def log_dataset(self, images, labels, class_names):
        """Log dataset information"""
        logger.info("=" * 80)
        logger.info("DATASET LOADING")
        logger.info("=" * 80)
        
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip([class_names[i] for i in unique], counts))
        
        self.logs['dataset'] = {
            'total_images': len(images),
            'image_shape': images[0].shape,
            'num_classes': len(class_names),
            'class_names': class_names,
            'class_distribution': class_distribution,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✓ Loaded {len(images)} images")
        logger.info(f"✓ Image shape: {images[0].shape}")
        logger.info(f"✓ Classes: {class_names}")
        logger.info(f"✓ Distribution: {class_distribution}")
        
    def log_preprocessing(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Log preprocessing and data split"""
        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPROCESSING & SPLITTING")
        logger.info("=" * 80)
        
        self.logs['preprocessing'] = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape,
            'preprocessing_steps': [
                'CLAHE enhancement',
                'Gamma correction',
                'Resizing to 224x224',
                'Normalization'
            ],
            'augmentation': [
                'Rotation (±20°)',
                'Width/Height shift (10%)',
                'Shear (20%)',
                'Zoom (20%)',
                'Horizontal flip'
            ]
        }
        
        logger.info(f"✓ Train set: {len(X_train)} samples")
        logger.info(f"✓ Validation set: {len(X_val)} samples")
        logger.info(f"✓ Test set: {len(X_test)} samples")
        logger.info(f"✓ Input shape: {X_train.shape[1:]}")
        
    def log_model_start(self, model_name, model_type='CNN'):
        """Log the start of model training"""
        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING: {model_name.upper()} ({model_type})")
        logger.info("=" * 80)
        
    def log_model_results(self, model_name, results, training_time):
        """Log model training results"""
        self.logs['models'][model_name] = {
            'training_time': training_time,
            'metrics': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"\n✓ {model_name} completed in {training_time:.2f}s")
        logger.info(f"  - Test Accuracy: {results.get('accuracy', 0):.4f}")
        logger.info(f"  - Test Loss: {results.get('loss', 0):.4f}")
        
    def save_logs(self):
        """Save all logs to JSON file"""
        log_file = self.output_dir / f'comprehensive_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=4, default=str)
        logger.info(f"\n✓ Logs saved to {log_file}")
        return log_file


def train_cnn_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, 
                    class_weights, comp_logger):
    """Train and evaluate a single CNN model"""
    comp_logger.log_model_start(model_name, 'CNN')
    start_time = time.time()
    
    try:
        # Create model
        logger.info(f"Creating {model_name} architecture...")
        pretrained = PretrainedModels(
            model_name=model_name,
            input_shape=DATA_CONFIG['image_size'] + (3,),
            num_classes=DATA_CONFIG['num_classes'],
            freeze_base=MODEL_CONFIG['freeze_base']
        )
        model = pretrained.get_model()
        
        # Compile model
        logger.info("Compiling model...")
        model.compile(
            optimizer=TRAINING_CONFIG['optimizer'],
            loss=TRAINING_CONFIG['loss'],
            metrics=TRAINING_CONFIG['metrics']
        )
        
        # Train model
        logger.info("Starting training...")
        trainer = Trainer(model, model_name=model_name, save_dir=f'results/{model_name}')
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            class_weights=class_weights
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, class_names=DATA_CONFIG['class_names'])
        results = evaluator.evaluate(X_test, y_test)
        
        training_time = time.time() - start_time
        comp_logger.log_model_results(model_name, results, training_time)
        
        return {
            'model_name': model_name,
            'model_type': 'CNN',
            'results': results,
            'history': history.history if hasattr(history, 'history') else history,
            'training_time': training_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"✗ Error training {model_name}: {str(e)}")
        training_time = time.time() - start_time
        return {
            'model_name': model_name,
            'model_type': 'CNN',
            'error': str(e),
            'training_time': training_time,
            'success': False
        }


def train_vit_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                    class_weights, comp_logger):
    """Train and evaluate Vision Transformer"""
    model_name = 'vision_transformer'
    comp_logger.log_model_start(model_name, 'Vision Transformer')
    start_time = time.time()
    
    try:
        # Create ViT model
        logger.info("Creating Vision Transformer architecture...")
        from src.models.vision_transformer import create_vit_model
        model = create_vit_model(
            image_size=224,
            patch_size=16,
            num_classes=DATA_CONFIG['num_classes'],
            depth=12,
            dim=768,
            mlp_dim=3072,
            heads=12,
            dropout=0.1
        )
        
        # Compile model
        logger.info("Compiling model...")
        model.compile(
            optimizer=TRAINING_CONFIG['optimizer'],
            loss=TRAINING_CONFIG['loss'],
            metrics=TRAINING_CONFIG['metrics']
        )
        
        # Train model
        logger.info("Starting training...")
        trainer = Trainer(model, model_name=model_name, save_dir=f'results/{model_name}')
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            class_weights=class_weights
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, class_names=DATA_CONFIG['class_names'])
        results = evaluator.evaluate(X_test, y_test)
        
        training_time = time.time() - start_time
        comp_logger.log_model_results(model_name, results, training_time)
        
        return {
            'model_name': model_name,
            'model_type': 'Vision Transformer',
            'results': results,
            'history': history.history if hasattr(history, 'history') else history,
            'training_time': training_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"✗ Error training Vision Transformer: {str(e)}")
        training_time = time.time() - start_time
        return {
            'model_name': model_name,
            'model_type': 'Vision Transformer',
            'error': str(e),
            'training_time': training_time,
            'success': False
        }


def create_comprehensive_comparison(all_results, output_dir):
    """Create comprehensive comparison visualizations and reports"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("CREATING COMPREHENSIVE COMPARISON")
    logger.info("=" * 80)
    
    # Filter successful results
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        logger.error("No successful model results to compare!")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in successful_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Type': result['model_type'],
            'Test Accuracy': result['results'].get('accuracy', 0),
            'Test Loss': result['results'].get('loss', 0),
            'Precision': result['results'].get('precision', 0),
            'Recall': result['results'].get('recall', 0),
            'F1-Score': result['results'].get('f1_score', 0),
            'Training Time (s)': result['training_time']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Test Accuracy', ascending=False)
    
    # Save comparison CSV
    csv_file = output_dir / 'model_comparison.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"✓ Comparison table saved to {csv_file}")
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    print("\n" + df.to_string(index=False))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Test Accuracy Comparison
    ax = axes[0, 0]
    colors = ['#2ecc71' if t == 'Vision Transformer' else '#3498db' for t in df['Type']]
    ax.barh(df['Model'], df['Test Accuracy'], color=colors)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Training Time Comparison
    ax = axes[0, 1]
    ax.barh(df['Model'], df['Training Time (s)'], color=colors)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. F1-Score Comparison
    ax = axes[0, 2]
    ax.barh(df['Model'], df['F1-Score'], color=colors)
    ax.set_xlabel('F1-Score')
    ax.set_title('F1-Score Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Accuracy vs Training Time Scatter
    ax = axes[1, 0]
    for model_type in df['Type'].unique():
        subset = df[df['Type'] == model_type]
        ax.scatter(subset['Training Time (s)'], subset['Test Accuracy'], 
                  label=model_type, s=100, alpha=0.6)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Training Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Multi-metric Comparison (Top 5 models)
    ax = axes[1, 1]
    top_5 = df.head(5)
    x = np.arange(len(top_5))
    width = 0.2
    ax.bar(x - width, top_5['Test Accuracy'], width, label='Accuracy')
    ax.bar(x, top_5['Precision'], width, label='Precision')
    ax.bar(x + width, top_5['Recall'], width, label='Recall')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Metric Comparison (Top 5)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_5['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Model Type Summary
    ax = axes[1, 2]
    type_summary = df.groupby('Type').agg({
        'Test Accuracy': 'mean',
        'Training Time (s)': 'mean'
    }).reset_index()
    x = np.arange(len(type_summary))
    ax.bar(x, type_summary['Test Accuracy'], color=['#3498db', '#2ecc71'])
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Average Test Accuracy')
    ax.set_title('Average Performance by Model Type')
    ax.set_xticks(x)
    ax.set_xticklabels(type_summary['Type'])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'comprehensive_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Comparison visualization saved to {plot_file}")
    plt.close()
    
    # Print winner
    logger.info("\n" + "=" * 80)
    logger.info("BEST PERFORMING MODELS")
    logger.info("=" * 80)
    logger.info(f"🏆 Best Accuracy: {df.iloc[0]['Model']} ({df.iloc[0]['Test Accuracy']:.4f})")
    logger.info(f"⚡ Fastest Training: {df.loc[df['Training Time (s)'].idxmin(), 'Model']} "
                f"({df['Training Time (s)'].min():.2f}s)")
    logger.info(f"📊 Best F1-Score: {df.loc[df['F1-Score'].idxmax(), 'Model']} "
                f"({df['F1-Score'].max():.4f})")
    
    return df


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL COMPARISON")
    logger.info("Training ALL CNN Models + Vision Transformer")
    logger.info("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'results/comprehensive_comparison_{timestamp}')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize comprehensive logger
    comp_logger = ComprehensiveLogger(output_dir)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading Dataset...")
    data_loader = DataLoader(
        data_dir=DATA_CONFIG['raw_data_dir'],
        class_names=DATA_CONFIG['class_names']
    )
    images, labels = data_loader.load_images_from_directory(
        DATA_CONFIG['raw_data_dir']
    )
    comp_logger.log_dataset(images, labels, DATA_CONFIG['class_names'])
    
    # Step 2: Preprocess data
    logger.info("\nStep 2: Preprocessing Images...")
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8)
    )
    processed_images = preprocessor.preprocess_batch(images)
    
    # Step 3: Split data
    logger.info("\nStep 3: Splitting Dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data(
        processed_images, labels,
        test_size=DATA_CONFIG['test_size'],
        val_size=DATA_CONFIG['val_size']
    )
    comp_logger.log_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Step 4: Handle class imbalance
    logger.info("\nStep 4: Calculating Class Weights...")
    imbalance_handler = ClassImbalanceHandler()
    class_weights = imbalance_handler.compute_class_weights(y_train)
    logger.info(f"✓ Class weights: {class_weights}")
    
    # Step 5: Train all models
    logger.info("\n" + "=" * 80)
    logger.info(f"Step 5: Training {len(CNN_MODELS) + 1} Models")
    logger.info("=" * 80)
    
    all_results = []
    
    # Train all CNN models
    for i, model_name in enumerate(CNN_MODELS, 1):
        logger.info(f"\n[{i}/{len(CNN_MODELS) + 1}] Training {model_name}...")
        result = train_cnn_model(
            model_name, X_train, y_train, X_val, y_val, X_test, y_test,
            class_weights, comp_logger
        )
        all_results.append(result)
    
    # Train Vision Transformer
    logger.info(f"\n[{len(CNN_MODELS) + 1}/{len(CNN_MODELS) + 1}] Training Vision Transformer...")
    vit_result = train_vit_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        class_weights, comp_logger
    )
    all_results.append(vit_result)
    
    # Step 6: Create comprehensive comparison
    logger.info("\nStep 6: Creating Comprehensive Comparison...")
    comparison_df = create_comprehensive_comparison(all_results, output_dir)
    
    # Step 7: Save logs
    logger.info("\nStep 7: Saving Comprehensive Logs...")
    log_file = comp_logger.save_logs()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE COMPARISON COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✓ Results saved to: {output_dir}")
    logger.info(f"✓ Comparison CSV: {output_dir}/model_comparison.csv")
    logger.info(f"✓ Visualizations: {output_dir}/comprehensive_comparison.png")
    logger.info(f"✓ Detailed logs: {log_file}")
    logger.info(f"✓ Individual model results: results/[model_name]/")
    logger.info("\n🎉 All done! Check the results directory for detailed analysis.")


if __name__ == '__main__':
    main()
