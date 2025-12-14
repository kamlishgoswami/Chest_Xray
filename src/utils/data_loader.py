"""
Data loading utilities
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class DataLoader:
    """
    Utility class for loading and preparing chest X-ray datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing image data
            class_names: List of class names (subdirectory names)
        """
        self.data_dir = data_dir
        
        if class_names is None:
            self.class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def load_images_from_directory(
        self,
        directory: str,
        target_size: Tuple[int, int] = (224, 224),
        grayscale: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from directory structure.
        
        Expected structure:
        directory/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
        
        Args:
            directory: Path to data directory
            target_size: Target size for images
            grayscale: Whether to load as grayscale
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(directory, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping.")
                continue
            
            print(f"Loading images from {class_name}...")
            
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                
                # Load image
                if grayscale:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img is None:
                    print(f"Warning: Failed to load {img_path}")
                    continue
                
                # Resize
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
                
                images.append(img)
                labels.append(class_idx)
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images across {self.num_classes} classes")
        
        return images, labels
    
    def prepare_data(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        categorical: bool = True
    ) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            images: Image array
            labels: Label array
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)
            random_state: Random seed
            categorical: Whether to convert labels to categorical
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Convert to categorical if needed
        if categorical:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_val = to_categorical(y_val, num_classes=self.num_classes)
            y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def save_processed_data(
        save_path: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Save processed data to disk.
        
        Args:
            save_path: Directory to save data
            X_train, X_val, X_test: Image arrays
            y_train, y_val, y_test: Label arrays
        """
        os.makedirs(save_path, exist_ok=True)
        
        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_val.npy'), X_val)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_val.npy'), y_val)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        
        print(f"Data saved to {save_path}")
    
    @staticmethod
    def load_processed_data(load_path: str) -> Tuple:
        """
        Load processed data from disk.
        
        Args:
            load_path: Directory containing saved data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X_train = np.load(os.path.join(load_path, 'X_train.npy'))
        X_val = np.load(os.path.join(load_path, 'X_val.npy'))
        X_test = np.load(os.path.join(load_path, 'X_test.npy'))
        y_train = np.load(os.path.join(load_path, 'y_train.npy'))
        y_val = np.load(os.path.join(load_path, 'y_val.npy'))
        y_test = np.load(os.path.join(load_path, 'y_test.npy'))
        
        print(f"Data loaded from {load_path}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
