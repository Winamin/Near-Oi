"""
CAR System - Compare-Adjust-Record Computational Architecture

A novel computational architecture for emergent pattern detection
through iterative computational unit interactions.
"""

from .car_system import CARSystem
from .enhanced_car import EnhancedCARSystem
from .qm9_dataset import QM9Dataset, generate_molecular_features
from .experiment import run_qm9_experiment, run_comparison_experiments

__all__ = [
    'CARSystem',
    'EnhancedCARSystem',
    'QM9Dataset',
    'generate_molecular_features',
    'run_qm9_experiment',
    'run_comparison_experiments',
]

__version__ = '1.0.0'
__author__ = 'Winamin'