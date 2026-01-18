import numpy as np
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_car import EnhancedCARSystem
from src.qm9_dataset import QM9Dataset


def run_qm9_experiment():
    """Run complete experiment with QM9 dataset"""
    print("\n" + "="*80)
    print("CAR System Experiment with QM9 Dataset")
    print("="*80)
    
    # Initialize QM9 dataset handler
    qm9 = QM9Dataset()
    stats = qm9.get_qm9_statistics()
    
    print(f"\nQM9 Dataset Statistics:")
    print(f"  Mean HOMO-LUMO gap: {stats['mean']:.4f} eV")
    print(f"  Std: {stats['std']:.4f} eV")
    print(f"  Feature dimension: {stats['feature_dim']}")
    
    # Generate QM9-like data
    print(f"\nGenerating QM9-like dataset...")
    X, y = qm9.generate_synthetic_molecules(3000)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"HOMO-LUMO gap range: [{y.min():.2f}, {y.max():.2f}] eV")
    print(f"Mean: {y.mean():.2f} eV")
    
    # Create enhanced CAR system with paper parameters
    print(f"\nInitializing Enhanced CAR System...")
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
        exploration_value=stats['mean']  # Use QM9 mean
    )
    
    print(f"System initialized with paper parameters")
    
    # Run inference
    print(f"\nRunning inference on {len(X)} samples...")
    predictions = []
    errors = []
    knowledge_sizes = []
    special_pattern_sizes = []
    strategies = []
    
    for i, (features, target) in enumerate(zip(X, y)):
        result = car.infer(features, target)
        predictions.append(result['prediction'])
        error = abs(result['prediction'] - target)
        errors.append(error)
        knowledge_sizes.append(result['knowledge_size'])
        special_pattern_sizes.append(result['special_patterns_size'])
        strategies.append(result['strategy'])
        
        if (i + 1) % 500 == 0:
            recent_mae = np.mean(errors[-500:])
            recent_kb = knowledge_sizes[-1]
            recent_sp = special_pattern_sizes[-1]
            print(f"  {i+1}/{len(X)}: MAE={recent_mae:.4f} eV, "
                  f"KB={recent_kb}, SP={recent_sp}")
    
    predictions = np.array(predictions)
    errors = np.array(errors)
    
    # Compute metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Strategy statistics
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    stats = car.get_statistics()
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nPerformance metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} eV")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} eV")
    print(f"  R²: {r2:.4f}")
    
    print(f"\nStrategy usage:")
    for s, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} ({count/len(strategies)*100:.1f}%)")
    
    print(f"\nKnowledge base:")
    print(f"  Final size: {stats['knowledge_base_size']}")
    print(f"  Special patterns: {stats['special_patterns_size']}")
    print(f"  Patterns added: {stats['patterns_added']}")
    print(f"  Special patterns added: {stats['special_patterns_added']}")
    print(f"  Patterns merged: {stats['patterns_merged']}")
    
    print(f"\nSystem status:")
    print(f"  Current learning rate: {stats['current_learning_rate']:.4f}")
    print(f"  Recent error: {stats['recent_error']:.4f} eV")
    
    # Compare with paper results
    print(f"\n" + "="*80)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*80)
    print(f"Paper MAE: 1.07 eV")
    print(f"Our MAE: {mae:.4f} eV")
    print(f"Improvement: {((1.07 - mae) / 1.07 * 100):.1f}%")
    
    if mae <= 1.07:
        print(f"✓ Our implementation achieves paper-level performance!")
    else:
        print(f"✗ Our implementation achieves {((1.07 - mae) / 1.07 * 100):.1f}% worse performance")
        print(f"  Further optimization needed")
    
    # Additional analysis
    print(f"\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Knowledge accumulation
    print(f"Knowledge accumulation:")
    print(f"  Total patterns: {stats['patterns_added'] + stats['special_patterns_added']}")
    print(f"  Knowledge patterns: {stats['patterns_added']}")
    print(f"  Special patterns: {stats['special_patterns_added']}")
    
    # Coverage analysis
    coverage = len([e for e in errors if e < 1.0]) / len(errors) * 100
    print(f"  Coverage (error < 1.0 eV): {coverage:.1f}%")
    
    # Consensus analysis
    consensus_rate = stats['consensus_reached'] / stats['total_inferences'] * 100
    print(f"  Consensus rate: {consensus_rate:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'errors': errors,
        'knowledge_sizes': knowledge_sizes,
        'special_pattern_sizes': special_pattern_sizes,
        'strategy_counts': strategy_counts,
        'statistics': stats
    }


def run_comparison_experiments():
    """Run comparison between different implementations"""
    print("\n" + "="*80)
    print("COMPARISON BETWEEN DIFFERENT IMPLEMENTATIONS")
    print("="*80)
    
    # Generate QM9-like data
    qm9 = QM9Dataset()
    X, y = qm9.generate_synthetic_molecules(3000)
    
    results = {}
    
    # 1. Basic fixed weights (no learning)
    print("\n[1] Basic fixed-weight CAR (no learning)...")
    car_basic = EnhancedCARSystem(
        num_units=20, feature_dim=69, kb_capacity=1,
        learning_rate=0.0, consensus_threshold=0.95,
        similarity_thresholds=[0.99], pattern_merge_threshold=0.99,
        reflection_interval=999999, success_threshold=1.0,
        exploration_value=7.5
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_basic.infer(f)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['basic'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2))
    }
    print(f"  MAE: {results['basic']['mae']:.4f} eV")
    
    # 2. Knowledge base CAR
    print("\n[2] Knowledge base CAR...")
    car_kb = EnhancedCARSystem(
        num_units=10, feature_dim=69, kb_capacity=500,
        learning_rate=0.2, consensus_threshold=0.8,
        similarity_thresholds=[0.5], pattern_merge_threshold=0.85,
        reflection_interval=50, success_threshold=1.0,
        exploration_value=7.5
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_kb.infer(f, t)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['knowledge'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2)),
        'kb_size': len(car_kb.knowledge_base),
        'special_patterns': len(car_kb.special_patterns)
    }
    print(f"  MAE: {results['knowledge']['mae']:.4f} eV, KB: {results['knowledge']['kb_size']}, SP: {results['knowledge']['special_patterns']}")
    
    # 3. Full enhanced CAR system (paper-level)
    print("\n[3] Enhanced CAR system (paper-level)...")
    car_enhanced = EnhancedCARSystem(
        num_units=20, feature_dim=69, kb_capacity=2000,  # Paper values
        learning_rate=0.3, consensus_threshold=0.6,
        similarity_thresholds=[0.2, 0.4, 0.6],  # Paper values
        pattern_merge_threshold=0.70,  # Paper value
        special_pattern_threshold=0.25,  # Paper value
        diversity_bonus_factor=0.20,  # Paper value
        reflection_interval=30, success_threshold=1.0,
        exploration_value=7.5
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_enhanced.infer(f, t)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['enhanced'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2)),
        'kb_size': len(car_enhanced.knowledge_base),
        'special_patterns': len(car_enhanced.special_patterns),
        'stats': car_enhanced.get_statistics()
    }
    print(f"  MAE: {results['enhanced']['mae']:.4f} eV, KB: {results['enhanced']['kb_size']}, SP: {results['enhanced']['special_patterns']}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Method':<25} {'MAE (eV)':<15} {'RMSE (eV)':<15} {'KB':<8} {'SP':<8}")
    print("-" * 75)
    for name, res in results.items():
        kb_size = res.get('kb_size', 'N/A')
        sp_size = res.get('special_patterns', 'N/A')
        print(f"{name:<25} {res['mae']:<15.4f} {res['rmse']:<15.4f} {str(kb_size):<8} {str(sp_size):<8}")
    
    return results


if __name__ == "__main__":
    try:
        # Run main experiment
        main_results = run_qm9_experiment()
        
        # Run comparison
        comparison_results = run_comparison_experiments()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        
        # Final summary
        print(f"\nFINAL SUMMARY:")
        print(f"  Enhanced CAR MAE: {main_results['mae']:.4f} eV")
        print(f"  Paper MAE: 1.07 eV")
        print(f"  Performance: {'✓ PASS' if main_results['mae'] <= 1.07 else '✗ FAIL'}")
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()