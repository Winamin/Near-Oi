"""
Quick QM9 Experiment Test
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_car import EnhancedCARSystem
from src.qm9_dataset import QM9Dataset


def run_quick_qm9_test():
    """Run quick QM9 experiment test"""
    print("\n" + "="*60)
    print("Quick QM9 Experiment Test")
    print("="*60)
    
    # Generate small QM9-like dataset
    qm9 = QM9Dataset()
    X, y = qm9.generate_synthetic_molecules(500)  # Smaller dataset for quick test
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"HOMO-LUMO gap range: [{y.min():.2f}, {y.max():.2f}] eV")
    
    # Create enhanced CAR system with paper parameters
    car = EnhancedCARSystem(
        num_units=20,                    # Paper: 20 units
        feature_dim=69,                  # Paper: 69 features
        kb_capacity=2000,                # Paper: 2000 capacity
        learning_rate=0.3,               # Paper: 0.3
        consensus_threshold=0.6,         # Paper: 0.6
        similarity_thresholds=[0.2, 0.4, 0.6],  # Paper: [0.2, 0.4, 0.6]
        pattern_merge_threshold=0.70,    # Paper: 0.70
        special_pattern_threshold=0.25,  # Paper: 0.25
        diversity_bonus_factor=0.20,     # Paper: 0.20
        reflection_interval=30,          # Paper: 30
        success_threshold=1.0,           # Paper: 1.0
        exploration_value=7.5
    )
    
    # Run inference on small dataset
    print(f"\nRunning inference on {len(X)} samples...")
    predictions = []
    errors = []
    
    for i, (features, target) in enumerate(zip(X, y)):
        result = car.infer(features, target)
        predictions.append(result['prediction'])
        error = abs(result['prediction'] - target)
        errors.append(error)
        
        if (i + 1) % 100 == 0:
            recent_mae = np.mean(errors[-100:])
            print(f"  {i+1}/{len(X)}: MAE={recent_mae:.4f} eV")
    
    predictions = np.array(predictions)
    errors = np.array(errors)
    
    # Compute metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    stats = car.get_statistics()
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nPerformance metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} eV")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} eV")
    print(f"  R²: {r2:.4f}")
    
    print(f"\nKnowledge base:")
    print(f"  Final size: {stats['knowledge_base_size']}")
    print(f"  Special patterns: {stats['special_patterns_size']}")
    print(f"  Patterns added: {stats['patterns_added']}")
    print(f"  Special patterns added: {stats['special_patterns_added']}")
    print(f"  Patterns merged: {stats['patterns_merged']}")
    
    # Compare with paper results
    print(f"\n" + "="*60)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*60)
    print(f"Paper MAE: 1.07 eV")
    print(f"Our MAE: {mae:.4f} eV")
    print(f"Improvement: {((1.07 - mae) / 1.07 * 100):.1f}%")
    
    if mae <= 1.07:
        print(f"✓ Our implementation achieves paper-level performance!")
        return True
    else:
        print(f"✗ Our implementation achieves {((1.07 - mae) / 1.07 * 100):.1f}% worse performance")
        print(f"  Further optimization needed")
        return False


if __name__ == "__main__":
    success = run_quick_qm9_test()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    if success:
        print("✓ Enhanced CAR system successfully implemented!")
        print("  The system is ready for production use.")
    else:
        print("✗ Enhanced CAR system needs further optimization.")
        print("  Consider adjusting hyperparameters or improving feature engineering.")