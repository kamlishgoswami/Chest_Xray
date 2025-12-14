"""
Image preprocessing pipeline for chest X-ray images
Includes resizing, CLAHE, gamma correction, and super-resolution
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Image preprocessing pipeline for chest X-ray images.
    
    Features:
    - Resizing to 224×224 for compatibility with pretrained models
    - CLAHE for local contrast enhancement
    - Gamma correction for brightness adjustment
    - Optional super-resolution
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        apply_gamma: bool = False,
        gamma_value: float = 1.2,
        normalize: bool = True
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            apply_clahe: Whether to apply CLAHE
            clahe_clip_limit: CLAHE clip limit for histogram equalization
            clahe_tile_size: Tile size for CLAHE algorithm
            apply_gamma: Whether to apply gamma correction
            gamma_value: Gamma value for correction (>1 brightens, <1 darkens)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.apply_gamma = apply_gamma
        self.gamma_value = gamma_value
        self.normalize = normalize
        
        # Initialize CLAHE
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Suitable for medical imaging with tile-based histogram equalization.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for RGB images
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Apply directly for grayscale images
            enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Apply gamma correction for brightness and luminance adjustment.
        
        Args:
            image: Input image
            gamma: Gamma value (if None, uses self.gamma_value)
            
        Returns:
            Gamma corrected image
        """
        if gamma is None:
            gamma = self.gamma_value
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using lookup table
        return cv2.LUT(image, table)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input image (can be grayscale or RGB)
            
        Returns:
            Preprocessed image
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize to target size
        image = self.resize_image(image)
        
        # Apply CLAHE enhancement
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
        
        # Apply gamma correction
        if self.apply_gamma:
            image = self.apply_gamma_correction(image)
        
        # Normalize
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Preprocessed images as numpy array
        """
        preprocessed = [self.preprocess(img) for img in images]
        return np.array(preprocessed)


class SuperResolution:
    """
    Optional super-resolution module for low-resolution X-ray images.
    
    References:
    - SRCNN: Image Super-Resolution Using Deep Convolutional Networks
    - VDSR: Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    
    Note: This is a placeholder for integration with SR models.
    For actual implementation, pretrained models or custom training is required.
    """
    
    def __init__(self, scale_factor: int = 2, method: str = 'bicubic'):
        """
        Initialize super-resolution module.
        
        Args:
            scale_factor: Upscaling factor
            method: Upscaling method ('bicubic', 'srcnn', 'vdsr')
        """
        self.scale_factor = scale_factor
        self.method = method
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image using selected method.
        
        Args:
            image: Input low-resolution image
            
        Returns:
            Upscaled image
        """
        if self.method == 'bicubic':
            # Simple bicubic interpolation
            new_size = (image.shape[1] * self.scale_factor, 
                       image.shape[0] * self.scale_factor)
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        elif self.method in ['srcnn', 'vdsr']:
            # Placeholder for deep learning-based super-resolution
            # In practice, would load pretrained model and apply inference
            print(f"Warning: {self.method} not implemented. Using bicubic interpolation.")
            new_size = (image.shape[1] * self.scale_factor, 
                       image.shape[0] * self.scale_factor)
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError(f"Unknown super-resolution method: {self.method}")
