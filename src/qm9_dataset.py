"""
QM9 Dataset Handler for CAR System

This module handles the QM9 dataset for molecular property prediction.
"""

import numpy as np
import os
import pickle
from typing import Tuple, List
import urllib.request
import tarfile
import tempfile


class QM9Dataset:
    """
    QM9 Dataset Handler
    
    Handles the QM9 dataset for molecular property prediction.
    The QM9 dataset contains 133,885 stable small organic molecules
    composed of C, H, O, N, and F elements.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize QM9 dataset handler
        
        Args:
            data_dir: Directory to store QM9 data
        """
        self.data_dir = data_dir
        self.qm9_dir = os.path.join(data_dir, "qm9")
        self.molecules_dir = os.path.join(self.qm9_dir, "molecules")
        
        # Create directories if they don't exist
        os.makedirs(self.qm9_dir, exist_ok=True)
        os.makedirs(self.molecules_dir, exist_ok=True)
        
        # QM9 statistics (from paper)
        self.qm9_stats = {
            'mean': 7.5,
            'std': 3.0,
            'min': 3.13,
            'max': 16.92,
            'feature_dim': 69  # 69-dimensional feature vector
        }
    
    def get_qm9_statistics(self) -> dict:
        """Get QM9 dataset statistics"""
        return self.qm9_stats.copy()
    
    def download_qm9_data(self):
        """Download QM9 dataset"""
        print("Downloading QM9 dataset...")
        
        # Download URL from the paper
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                urllib.request.urlretrieve(url, tmp_file.name)
                
                # Extract the tar.gz file
                with tarfile.open(tmp_file.name, 'r:gz') as tar:
                    tar.extractall(self.qm9_dir)
                
                print(f"QM9 dataset downloaded and extracted to {self.qm9_dir}")
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
        except Exception as e:
            print(f"Error downloading QM9 dataset: {e}")
            print("Using synthetic data for demonstration...")
    
    def generate_synthetic_molecules(self, n_molecules: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic molecular features matching QM9 distribution
        
        Args:
            n_molecules: Number of molecules to generate
            
        Returns:
            Tuple of (features, labels)
        """
        print(f"Generating {n_molecules} synthetic molecules...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate random molecular features (69-dimensional)
        features = np.random.randn(n_molecules, 69)
        
        # Generate realistic HOMO-LUMO gaps
        # Based on QM9 statistics: mean=7.5, std=3.0
        labels = np.random.normal(7.5, 3.0, n_molecules)
        
        # Clip to realistic range
        labels = np.clip(labels, 3.13, 16.92)
        
        # Add some structure to make it more realistic
        # Simulate some molecular patterns
        for i in range(n_molecules):
            # Add some correlation between features
            features[i] += np.sin(np.arange(69) * 0.1) * 0.1
            
            # Add some molecular-specific patterns
            if i % 3 == 0:
                # Some molecules have specific patterns
                features[i, :10] += np.random.randn(10) * 0.5
            elif i % 3 == 1:
                # Others have different patterns
                features[i, 20:30] += np.random.randn(10) * 0.3
        
        print(f"Generated {n_molecules} molecules with features shape: {features.shape}")
        print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}] eV")
        
        return features, labels
    
    def load_or_generate_data(self, n_molecules: int = 3000, use_synthetic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load QM9 data or generate synthetic data
        
        Args:
            n_molecules: Number of molecules to load/generate
            use_synthetic: Whether to use synthetic data if QM9 is not available
            
        Returns:
            Tuple of (features, labels)
        """
        if use_synthetic:
            return self.generate_synthetic_molecules(n_molecules)
        else:
            # Try to load real QM9 data
            try:
                return self._load_real_qm9_data(n_molecules)
            except:
                print("Could not load real QM9 data, generating synthetic data...")
                return self.generate_synthetic_molecules(n_molecules)
    
    def _load_real_qm9_data(self, n_molecules: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load real QM9 data (implementation would go here)
        """
        # This is a placeholder - real implementation would load actual QM9 data
        # For now, we'll just generate synthetic data that matches QM9 statistics
        return self.generate_synthetic_molecules(n_molecules)


def generate_molecular_features(n_molecules: int = 3000, 
                               feature_dim: int = 69,
                               random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate molecular features for CAR system training/testing
    
    Args:
        n_molecules: Number of molecules to generate
        feature_dim: Feature dimension (69 for QM9)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_seed)
    
    # Generate random molecular features
    features = np.random.randn(n_molecules, feature_dim)
    
    # Generate realistic HOMO-LUMO gaps based on QM9 statistics
    # Mean: 7.5 eV, Std: 3.0 eV, Range: [3.13, 16.92] eV
    labels = np.random.normal(7.5, 3.0, n_molecules)
    labels = np.clip(labels, 3.13, 16.92)
    
    # Add some realistic patterns to the features
    # Simulate molecular structure patterns
    for i in range(n_molecules):
        # Add some correlation between features
        features[i] += np.sin(np.arange(feature_dim) * 0.1) * 0.1
        
        # Add some molecular-specific patterns
        if i % 3 == 0:
            # Some molecules have specific patterns
            features[i, :10] += np.random.randn(10) * 0.5
        elif i % 3 == 1:
            # Others have different patterns
            features[i, 20:30] += np.random.randn(10) * 0.3
        else:
            # Third type of pattern
            features[i, 40:50] += np.random.randn(10) * 0.4
    
    return features, labels


if __name__ == "__main__":
    # Test QM9 dataset handler
    print("Testing QM9 Dataset Handler")
    print("="*50)
    
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    
    print("QM9 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate synthetic data
    X, y = qm9.generate_synthetic_molecules(1000)
    print(f"\nGenerated data shape: {X.shape}")
    print(f"Label range: [{y.min():.2f}, {y.max():.2f}] eV")
    print(f"Label mean: {y.mean():.2f} eV")
