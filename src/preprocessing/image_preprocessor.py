"""
Unified preprocessing module for chest X-ray images.
Handles resizing, normalization, and data augmentation.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


class ChestXrayPreprocessor:
    """
    Unified preprocessing class for chest X-ray images.
    Ensures all images are resized to 224x224 for compatibility with pretrained models.
    """
    
    def __init__(self, target_size=(224, 224), preprocessing_function=None):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Tuple of (height, width) for resizing images. Default is (224, 224).
            preprocessing_function: Optional preprocessing function for specific model requirements.
        """
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
    
    def create_train_generator(self, 
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'):
        """
        Create data augmentation generator for training data.
        
        Args:
            rotation_range: Degree range for random rotations
            width_shift_range: Fraction of total width for horizontal shifts
            height_shift_range: Fraction of total height for vertical shifts
            shear_range: Shear intensity (shear angle in degrees)
            zoom_range: Range for random zoom
            horizontal_flip: Boolean for random horizontal flips
            fill_mode: Points outside boundaries are filled according to mode
            
        Returns:
            ImageDataGenerator configured for training with augmentation
        """
        return ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
            preprocessing_function=self.preprocessing_function,
            rescale=1./255 if self.preprocessing_function is None else None
        )
    
    def create_validation_generator(self):
        """
        Create data generator for validation/test data without augmentation.
        
        Returns:
            ImageDataGenerator configured for validation without augmentation
        """
        return ImageDataGenerator(
            preprocessing_function=self.preprocessing_function,
            rescale=1./255 if self.preprocessing_function is None else None
        )
    
    def prepare_train_data(self, train_dir, batch_size=32, class_mode='categorical'):
        """
        Prepare training data with augmentation.
        
        Args:
            train_dir: Directory containing training images
            batch_size: Size of batches to generate
            class_mode: Type of label arrays (categorical, binary, etc.)
            
        Returns:
            Data generator for training
        """
        train_generator = self.create_train_generator()
        return train_generator.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=True
        )
    
    def prepare_validation_data(self, val_dir, batch_size=32, class_mode='categorical'):
        """
        Prepare validation data without augmentation.
        
        Args:
            val_dir: Directory containing validation images
            batch_size: Size of batches to generate
            class_mode: Type of label arrays (categorical, binary, etc.)
            
        Returns:
            Data generator for validation
        """
        val_generator = self.create_validation_generator()
        return val_generator.flow_from_directory(
            val_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=False
        )
    
    @staticmethod
    def get_preprocessing_function(model_type='vgg16'):
        """
        Get the appropriate preprocessing function for different model types.
        
        Args:
            model_type: Type of model ('vgg16', 'resnet50', 'efficientnet', etc.)
            
        Returns:
            Preprocessing function for the specified model
        """
        preprocessing_map = {
            'vgg16': vgg_preprocess,
            'vgg19': vgg_preprocess,
            'resnet50': resnet_preprocess,
            'resnet101': resnet_preprocess,
            'efficientnet': efficientnet_preprocess,
            'efficientnetb0': efficientnet_preprocess,
            'efficientnetb1': efficientnet_preprocess,
        }
        return preprocessing_map.get(model_type.lower(), None)
