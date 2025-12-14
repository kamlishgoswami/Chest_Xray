"""
Inference script for chest X-ray classification.
Load a trained model and make predictions on new images.
"""

import sys
import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.image_preprocessor import ChestXrayPreprocessor
from src.utils.explainer import ModelExplainer
from config.config import CLASS_NAMES, MODEL_CONFIG


def load_and_preprocess_image(image_path, target_size=(224, 224), model_type='resnet50'):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        model_type: Type of model for preprocessing
        
    Returns:
        Preprocessed image array
    """
    # Load image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply model-specific preprocessing
    preprocessing_func = ChestXrayPreprocessor.get_preprocessing_function(model_type)
    if preprocessing_func:
        img_array = preprocessing_func(img_array)
    else:
        img_array = img_array / 255.0
    
    return img_array


def predict_image(model, image_path, model_type='resnet50', generate_explanation=True):
    """
    Make prediction on a single image.
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
        model_type: Type of model used
        generate_explanation: Whether to generate Grad-CAM visualization
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_array = load_and_preprocess_image(
        image_path, 
        target_size=MODEL_CONFIG['input_shape'][:2],
        model_type=model_type
    )
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Print results
    print(f"\nPrediction Results for: {os.path.basename(image_path)}")
    print("-" * 60)
    print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll Class Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {predictions[0][i]:.2%}")
    print("-" * 60)
    
    # Generate explanation
    if generate_explanation:
        explainer = ModelExplainer(model)
        output_path = f"outputs/plots/explanation_{os.path.basename(image_path)}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result = explainer.visualize_prediction(
            img_array,
            class_names=CLASS_NAMES,
            save_path=output_path
        )
        print(f"\nExplanation visualization saved to: {output_path}")
    
    return {
        'predicted_class': CLASS_NAMES[predicted_class],
        'confidence': float(confidence),
        'all_predictions': {CLASS_NAMES[i]: float(predictions[0][i]) 
                          for i in range(len(CLASS_NAMES))}
    }


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(
        description='Make predictions on chest X-ray images'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs/models/chest_xray_classifier_best.h5',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='Path to the image file to predict'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='resnet50',
        help='Type of model (for preprocessing)'
    )
    parser.add_argument(
        '--no-explanation',
        action='store_true',
        help='Disable Grad-CAM explanation generation'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    print("=" * 60)
    print("Chest X-ray Classification - Inference")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Image: {args.image_path}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path)
    
    # Make prediction
    result = predict_image(
        model,
        args.image_path,
        model_type=args.model_type,
        generate_explanation=not args.no_explanation
    )
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
