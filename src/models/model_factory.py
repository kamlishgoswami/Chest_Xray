"""
Model factory for easy model creation
"""

from .pretrained_models import PretrainedModels, VisionTransformer
from typing import Tuple, Optional, List
from tensorflow.keras import Model


class ModelFactory:
    """
    Factory class for creating models with consistent interface.
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        model_name: Optional[str] = None,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 4,
        **kwargs
    ) -> Model:
        """
        Create a model based on type and name.
        
        Args:
            model_type: Type of model ('cnn' or 'vit')
            model_name: Name of the model (for CNN: 'resnet50', 'vgg16', etc.)
            input_shape: Input shape
            num_classes: Number of output classes
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Keras Model
        """
        if model_type.lower() == 'cnn':
            if model_name is None:
                raise ValueError("model_name is required for CNN models")
            
            pretrained = PretrainedModels(
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
                **kwargs
            )
            return pretrained.get_model()
        
        elif model_type.lower() == 'vit':
            vit = VisionTransformer(
                image_size=input_shape[0],
                num_classes=num_classes,
                channels=input_shape[2],
                **kwargs
            )
            return vit.get_model()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def list_available_models() -> dict:
        """
        List all available models.
        
        Returns:
            Dictionary of available models by type
        """
        return {
            'cnn': list(PretrainedModels.AVAILABLE_MODELS.keys()),
            'vit': ['vision_transformer']
        }
    
    @staticmethod
    def create_ensemble(
        model_names: List[str],
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 4,
        **kwargs
    ) -> List[Model]:
        """
        Create an ensemble of models.
        
        Args:
            model_names: List of model names
            input_shape: Input shape
            num_classes: Number of output classes
            **kwargs: Additional arguments
            
        Returns:
            List of models
        """
        models = []
        for model_name in model_names:
            if model_name == 'vit':
                model = ModelFactory.create_model(
                    'vit',
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **kwargs
                )
            else:
                model = ModelFactory.create_model(
                    'cnn',
                    model_name=model_name,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    **kwargs
                )
            models.append(model)
        
        return models
