"""
Denoising Autoencoder (DAE) for enhancing feature quality and robustness
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from typing import Tuple, Optional


class DenoisingAutoencoder:
    """
    Denoising Autoencoder trained with injected noise to enhance
    feature quality and robustness.
    
    Optimized using reconstruction loss and validation loss.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        latent_dim: int = 128,
        noise_factor: float = 0.2
    ):
        """
        Initialize Denoising Autoencoder.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            latent_dim: Dimension of latent space
            noise_factor: Factor for noise injection (0.0 to 1.0)
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        # Downsampling factor from three MaxPooling2D(2,2) layers: 2^3 = 8
        # This is critical for calculating the decoder's input shape
        self.downsampling_factor = 8
        self.model = None
        self.encoder = None
        self.decoder = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture."""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape, name='encoder_input')
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        
        # Flatten and bottleneck
        x = layers.Flatten()(x)
        encoder_output = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        # Create encoder model
        self.encoder = Model(encoder_input, encoder_output, name='encoder')
        
        # Decoder
        # Calculate the shape after encoding
        encoded_shape = (
            self.input_shape[0] // self.downsampling_factor,
            self.input_shape[1] // self.downsampling_factor,
            512
        )
        decoder_input = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        
        x = layers.Dense(np.prod(encoded_shape), activation='relu')(decoder_input)
        x = layers.Reshape(encoded_shape)(x)
        
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        decoder_output = layers.Conv2D(
            self.input_shape[2], (3, 3),
            activation='sigmoid',
            padding='same',
            name='decoder_output'
        )(x)
        
        # Create decoder model
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        
        # Full autoencoder
        autoencoder_input = encoder_input
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        
        self.model = Model(autoencoder_input, decoded, name='denoising_autoencoder')
    
    def add_noise(self, images: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to images.
        
        Args:
            images: Input images
            
        Returns:
            Noisy images
        """
        noise = np.random.normal(loc=0.0, scale=1.0, size=images.shape)
        noisy_images = images + self.noise_factor * noise
        noisy_images = np.clip(noisy_images, 0.0, 1.0)
        return noisy_images
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'mse'
    ):
        """
        Compile the autoencoder model.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function (mse or binary_crossentropy)
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['mae']
        )
    
    def train(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        save_path: Optional[str] = None
    ) -> keras.callbacks.History:
        """
        Train the denoising autoencoder.
        
        Args:
            X_train: Training images
            X_val: Validation images (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of callbacks
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        # Add noise to training data
        X_train_noisy = self.add_noise(X_train)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None:
            X_val_noisy = self.add_noise(X_val)
            validation_data = (X_val_noisy, X_val)
        
        # Default callbacks
        if callbacks is None:
            callbacks = []
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            )
            if save_path:
                callbacks.append(
                    ModelCheckpoint(
                        save_path,
                        monitor='val_loss' if X_val is not None else 'loss',
                        save_best_only=True,
                        verbose=1
                    )
                )
        
        # Train the model
        history = self.model.fit(
            X_train_noisy, X_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def denoise(self, images: np.ndarray) -> np.ndarray:
        """
        Denoise images using the trained autoencoder.
        
        Args:
            images: Input images (potentially noisy)
            
        Returns:
            Denoised images
        """
        return self.model.predict(images)
    
    def encode(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images to latent space.
        
        Args:
            images: Input images
            
        Returns:
            Latent representations
        """
        return self.encoder.predict(images)
    
    def decode(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Decode latent vectors to images.
        
        Args:
            latent_vectors: Latent representations
            
        Returns:
            Reconstructed images
        """
        return self.decoder.predict(latent_vectors)
    
    def save(self, path: str):
        """Save the autoencoder model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load the autoencoder model."""
        self.model = keras.models.load_model(path)
        # Reconstruct encoder and decoder
        self.encoder = Model(
            self.model.input,
            self.model.get_layer('latent').output,
            name='encoder'
        )
        # Note: decoder reconstruction may need adjustment based on architecture
    
    def get_model_summary(self):
        """Print model summaries."""
        print("=" * 80)
        print("ENCODER")
        print("=" * 80)
        self.encoder.summary()
        print("\n" + "=" * 80)
        print("DECODER")
        print("=" * 80)
        self.decoder.summary()
        print("\n" + "=" * 80)
        print("FULL AUTOENCODER")
        print("=" * 80)
        self.model.summary()
