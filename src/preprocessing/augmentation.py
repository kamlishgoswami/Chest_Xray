"""
Data augmentation module using ImageDataGenerator
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import Optional, Tuple


class DataAugmentation:
    """
    Data augmentation pipeline for chest X-ray images.
    
    Features:
    - Rotations (±20°)
    - Width/height shifts (10%)
    - Shear (20%)
    - Zoom (20%)
    - Horizontal flipping
    
    Improves generalization and handles class imbalance.
    """
    
    def __init__(
        self,
        rotation_range: float = 20.0,
        width_shift_range: float = 0.1,
        height_shift_range: float = 0.1,
        shear_range: float = 0.2,
        zoom_range: float = 0.2,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        fill_mode: str = 'nearest',
        preprocessing_function: Optional[callable] = None
    ):
        """
        Initialize data augmentation.
        
        Args:
            rotation_range: Degree range for random rotations (±20°)
            width_shift_range: Fraction of total width for horizontal shifts (10%)
            height_shift_range: Fraction of total height for vertical shifts (10%)
            shear_range: Shear intensity (20%)
            zoom_range: Range for random zoom (20%)
            horizontal_flip: Randomly flip inputs horizontally
            vertical_flip: Randomly flip inputs vertically
            fill_mode: Points outside boundaries are filled according to mode
            preprocessing_function: Optional preprocessing function
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self.preprocessing_function = preprocessing_function
        
        # Create ImageDataGenerator
        self.datagen = ImageDataGenerator(
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            fill_mode=self.fill_mode,
            preprocessing_function=self.preprocessing_function
        )
    
    def flow(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Generate batches of augmented data.
        
        Args:
            x: Input data (4D numpy array)
            y: Target data
            batch_size: Size of batches
            shuffle: Whether to shuffle data
            seed: Random seed
            
        Returns:
            Generator yielding (x_batch, y_batch)
        """
        return self.datagen.flow(
            x, y,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )
    
    def flow_from_directory(
        self,
        directory: str,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        class_mode: str = 'categorical',
        shuffle: bool = True,
        seed: Optional[int] = None,
        color_mode: str = 'rgb'
    ):
        """
        Generate batches of augmented data from directory.
        
        Args:
            directory: Path to directory
            target_size: Size to resize images to
            batch_size: Size of batches
            class_mode: Type of label arrays (categorical, binary, etc.)
            shuffle: Whether to shuffle data
            seed: Random seed
            color_mode: Color mode ('rgb' or 'grayscale')
            
        Returns:
            DirectoryIterator yielding batches
        """
        return self.datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed,
            color_mode=color_mode
        )
    
    def augment_image(self, image: np.ndarray, num_augmented: int = 1) -> np.ndarray:
        """
        Generate augmented versions of a single image.
        
        Args:
            image: Input image (3D or 4D numpy array)
            num_augmented: Number of augmented images to generate
            
        Returns:
            Array of augmented images
        """
        # Ensure image is 4D
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        augmented_images = []
        i = 0
        for batch in self.datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0])
            i += 1
            if i >= num_augmented:
                break
        
        return np.array(augmented_images)


class ClassImbalanceHandler:
    """
    Handle class imbalance in training data.
    """
    
    @staticmethod
    def compute_class_weights(y_train: np.ndarray) -> dict:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        if y_train.ndim > 1:
            # One-hot encoded labels
            y_train = np.argmax(y_train, axis=1)
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        return dict(zip(classes, weights))
    
    @staticmethod
    def oversample_minority_classes(
        X: np.ndarray,
        y: np.ndarray,
        augmenter: DataAugmentation,
        target_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority classes using augmentation.
        
        Args:
            X: Input data
            y: Labels
            augmenter: DataAugmentation instance
            target_samples: Target number of samples per class
            
        Returns:
            Oversampled (X, y)
        """
        if y.ndim > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        
        classes, counts = np.unique(y_labels, return_counts=True)
        
        if target_samples is None:
            target_samples = np.max(counts)
        
        X_balanced = []
        y_balanced = []
        
        for cls in classes:
            cls_indices = np.where(y_labels == cls)[0]
            cls_X = X[cls_indices]
            cls_y = y[cls_indices]
            
            X_balanced.append(cls_X)
            y_balanced.append(cls_y)
            
            # Generate additional samples if needed
            current_count = len(cls_indices)
            if current_count < target_samples:
                samples_needed = target_samples - current_count
                
                # Randomly select images to augment
                for _ in range(samples_needed):
                    idx = np.random.choice(len(cls_X))
                    augmented = augmenter.augment_image(cls_X[idx], num_augmented=1)
                    X_balanced.append(augmented)
                    y_balanced.append(cls_y[idx:idx+1])
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.vstack(y_balanced) if y.ndim > 1 else np.hstack(y_balanced)
        
        return X_balanced, y_balanced
