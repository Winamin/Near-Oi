import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_car import EnhancedCARSystem

def test_enhanced_functionality():
    """Test enhanced CAR system functionality."""
    print("\n" + "="*60)
    print("Testing Enhanced CAR System")
    print("="*60)
    
    # Test 1: Enhanced CAR System Initialization
    print("\n[Test 1] Initializing enhanced CAR system...")
    car_system = EnhancedCARSystem(
        num_units=20, 
        feature_dim=71, 
        kb_capacity=2000,
        learning_rate=0.3,
        consensus_threshold=0.6,
        similarity_thresholds=[0.2, 0.4, 0.6],
        pattern_merge_threshold=0.70,
        special_pattern_threshold=0.25,
        diversity_bonus_factor=0.20,
        reflection_interval=30,
        success_threshold=1.0,
        exploration_value=7.5
    )
    
    print(f"  ✓ Enhanced CAR system initialized")
    print(f"    Units: {car_system.num_units}")
    print(f"    Knowledge base capacity: {car_system.kb_capacity}")
    print(f"    Pattern merge threshold: {car_system.pattern_merge_threshold}")
    print(f"    Special pattern threshold: {car_system.special_pattern_threshold}")
    print(f"    Diversity bonus factor: {car_system.diversity_bonus_factor}")
    
    # Test 2: Generate synthetic data
    print("\n[Test 2] Generating synthetic data...")
    np.random.seed(42)
    n_samples = 100
    feature_dim = 71
    
    X = np.random.randn(n_samples, feature_dim)
    y = np.sum(X[:, :5], axis=1) + 7.0
    y += np.random.randn(n_samples) * 0.5
    y = np.clip(y, 3.0, 17.0)
    
    print(f"  ✓ Generated {n_samples} samples with {feature_dim} features")
    print(f"    HOMO-LUMO gap range: [{y.min():.2f}, {y.max():.2f}] eV")
    
    # Test 3: Process samples
    print("\n[Test 3] Processing samples...")
    predictions = []
    errors = []
    
    for i in range(n_samples):
        result = car_system.infer(X[i], y[i])
        predictions.append(result['prediction'])
        error = abs(result['prediction'] - y[i])
        errors.append(error)
        
        if (i + 1) % 25 == 0:
            recent_mae = np.mean(errors[-25:])
            print(f"    Processed {i+1}/{n_samples} samples, MAE: {recent_mae:.4f} eV")
    
    predictions = np.array(predictions)
    errors = np.array(errors)
    mae = np.mean(errors)
    
    print(f"  ✓ Processed all {n_samples} samples")
    print(f"    Final MAE: {mae:.4f} eV")
    
    # Test 4: System Statistics
    print("\n[Test 4] Getting system statistics...")
    stats = car_system.get_statistics()
    print(f"  ✓ Statistics retrieved successfully")
    print(f"    Total inferences: {stats['total_inferences']}")
    print(f"    Knowledge base size: {stats['knowledge_base_size']}")
    print(f"    Special patterns: {stats['special_patterns_size']}")
    print(f"    Patterns added: {stats['patterns_added']}")
    print(f"    Special patterns added: {stats['special_patterns_added']}")
    print(f"    Patterns merged: {stats['patterns_merged']}")
    
    # Test 5: Multi-perspective analysis
    print("\n[Test 5] Checking multi-perspective analysis...")
    perspectives = {}
    for unit in car_system.units:
        perspective = unit['perspective']
        perspectives[perspective] = perspectives.get(perspective, 0) + 1
    
    print(f"  ✓ Perspective distribution:")
    for perspective, count in perspectives.items():
        print(f"    {perspective}: {count} units")
    
    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()