"""
Data Augmentation Module

Supports multiple augmentation methods:
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Diffusion (Tabular Diffusion Model)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler


class DataAugmentation:
    """Data augmentation class supporting multiple methods."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.target_column = config['data'].get('target_column', 'Outcome')
        self.augmentation_method = config['data'].get('augmentation_method', 'diffusion')
        self.num_synthetic_samples = config['diffusion'].get('num_synthetic_samples', 200)
        
        # SMOTE/ADASYN parameters
        self.smote_k_neighbors = config['data'].get('smote_k_neighbors', 5)
        self.adasyn_n_neighbors = config['data'].get('adasyn_n_neighbors', 5)
        
        self.scaler = StandardScaler()
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray, 
                     feature_names: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation based on configured method.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Tuple of augmented (X, y)
        """
        self.logger.info(f"Applying {self.augmentation_method.upper()} data augmentation")
        
        if self.augmentation_method == 'smote':
            return self._apply_smote(X, y)
        elif self.augmentation_method == 'adasyn':
            return self._apply_adasyn(X, y)
        elif self.augmentation_method == 'diffusion':
            # Diffusion is handled separately in the main pipeline
            self.logger.info("Diffusion augmentation will be handled in separate step")
            return X, y
        elif self.augmentation_method == 'none':
            self.logger.info("No data augmentation applied")
            return X, y
        else:
            raise ValueError(f"Unknown augmentation method: {self.augmentation_method}")
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE augmentation."""
        self.logger.info("Applying SMOTE augmentation")
        
        # Calculate number of samples to generate
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        minority_count = np.min(counts)
        majority_count = np.max(counts)
        
        # Calculate sampling strategy to achieve balance
        sampling_strategy = {minority_class: majority_count}
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=min(self.smote_k_neighbors, minority_count - 1),
            random_state=self.config['data'].get('random_state', 42)
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        self.logger.info(f"SMOTE: Generated {len(X_resampled) - len(X)} synthetic samples")
        self.logger.info(f"SMOTE: Final class distribution - {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled
    
    def _apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN augmentation."""
        self.logger.info("Applying ADASYN augmentation")
        
        # Calculate number of samples to generate
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        minority_count = np.min(counts)
        majority_count = np.max(counts)
        
        # Calculate sampling strategy to achieve balance
        sampling_strategy = {minority_class: majority_count}
        
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=min(self.adasyn_n_neighbors, minority_count - 1),
            random_state=self.config['data'].get('random_state', 42)
        )
        
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        self.logger.info(f"ADASYN: Generated {len(X_resampled) - len(X)} synthetic samples")
        self.logger.info(f"ADASYN: Final class distribution - {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled
    
    def get_augmentation_stats(self, X_original: np.ndarray, y_original: np.ndarray,
                               X_augmented: np.ndarray, y_augmented: np.ndarray) -> dict:
        """Get statistics about the augmentation process."""
        stats = {
            'method': self.augmentation_method,
            'original_samples': len(X_original),
            'augmented_samples': len(X_augmented),
            'synthetic_samples': len(X_augmented) - len(X_original),
            'original_class_distribution': dict(zip(*np.unique(y_original, return_counts=True))),
            'augmented_class_distribution': dict(zip(*np.unique(y_augmented, return_counts=True)))
        }
        return stats
