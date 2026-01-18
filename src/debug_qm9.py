# Debug Real QM9 Experiment - Check for Data Issues

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_car import EnhancedCARSystem


def debug_qm9_data():
    """Debug the QM9 data loading and check for issues"""
    print("\n" + "="*80)
    print("DEBUGGING REAL QM9 DATA")
    print("="*80)
    
    # Path to QM9 CSV file
    csv_file = "data/gdb9.sdf.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: QM9 CSV file not found at {csv_file}")
        return
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Total samples in dataset: {len(df)}")
    print(f"Dataset columns: {list(df.columns)}")
    
    # Check for the gap column specifically
    if 'gap' in df.columns:
        print(f"\n'gap' column found:")
        print(f"  Min: {df['gap'].min():.4f}")
        print(f"  Max: {df['gap'].max():.4f}")
        print(f"  Mean: {df['gap'].mean():.4f}")
        print(f"  Std: {df['gap'].std():.4f}")
        
        # Show some sample values
        print(f"\nSample gap values:")
        print(df['gap'].head(10).values)
        
        # Check for constant values
        unique_gaps = df['gap'].unique()
        print(f"\nUnique gap values: {len(unique_gaps)}")
        if len(unique_gaps) <= 5:
            print("Warning: Very few unique gap values - might be constant!")
            print(f"Unique values: {unique_gaps}")
    else:
        print("\n'gap' column not found. Checking other columns...")
        # Look for any numeric column that might be the target
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns: {list(numeric_columns)}")
        
        # Check the first few rows of each numeric column
        for col in numeric_columns[:5]:
            print(f"\nColumn '{col}':")
            print(f"  Min: {df[col].min():.4f}")
            print(f"  Max: {df[col].max():.4f}")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std: {df[col].std():.4f}")
    
    # Check the actual data being loaded in the experiment
    print(f"\n" + "="*80)
    print("CHECKING ACTUAL DATA BEING USED")
    print("="*80)
    
    # Load data with the same parameters as the experiment
    df_sample = df.sample(n=10, random_state=42)
    print(f"Sample of 10 random molecules:")
    print(df_sample[['gap']].head(10))
    
    # Check if there's a systematic issue
    print(f"\nChecking for systematic patterns...")
    
    # Try to find molecules with very low gap values
    low_gap_molecules = df[df['gap'] < 5.0]
    print(f"Molecules with gap < 5.0 eV: {len(low_gap_molecules)}")
    
    # Try to find molecules with very high gap values
    high_gap_molecules = df[df['gap'] > 10.0]
    print(f"Molecules with gap > 10.0 eV: {len(high_gap_molecules)}")
    
    # Check if there's a clustering issue
    print(f"\nChecking gap distribution:")
    print(df['gap'].describe())
    
    # Check if the data is being processed correctly
    print(f"\n" + "="*80)
    print("CHECKING CAR SYSTEM DATA PROCESSING")
    print("="*80)
    
    # Try to load a small sample and run it through CAR
    X_sample, y_sample = load_real_qm9_data_debug(csv_file, n_samples=10)
    print(f"Sample data shape: {X_sample.shape}")
    print(f"Sample labels: {y_sample}")
    
    # Run a quick CAR inference on this sample
    car = EnhancedCARSystem(
        num_units=5,  # Small number for debugging
        feature_dim=X_sample.shape[1],
        kb_capacity=10,
        learning_rate=0.3,
        consensus_threshold=0.6,
        similarity_thresholds=[0.2, 0.4, 0.6],
        pattern_merge_threshold=0.70,
        special_pattern_threshold=0.25,
        diversity_bonus_factor=0.20,
        reflection_interval=10,
        success_threshold=1.0,
        exploration_value=np.mean(y_sample)
    )
    
    print(f"\nRunning CAR inference on sample data...")
    for i, (features, target) in enumerate(zip(X_sample, y_sample)):
        result = car.infer(features, target)
        error = abs(result['prediction'] - target)
        print(f"  Sample {i+1}: Target={target:.4f}, Prediction={result['prediction']:.4f}, Error={error:.4f}")
    
    print(f"\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


def load_real_qm9_data_debug(csv_file_path: str, n_samples: int = 10):
    """
    Debug version of load_real_qm9_data function
    """
    print(f"Loading real QM9 data from {csv_file_path}...")
    
    # Load CSV file
    df = pd.read_csv(csv_file_path)
    
    # Select random samples if n_samples is specified
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
        print(f"Selected {n_samples} random samples")
    
    # Look for molecular properties (HOMO-LUMO gap)
    property_columns = ['HOMO', 'LUMO', 'gap', 'homo_lumo_gap', 'gap_energy']
    
    target_column = None
    for col in property_columns:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            target_column = numeric_columns[0]
            print(f"Using first numeric column '{target_column}' as target")
        else:
            raise ValueError("No numeric columns found in the dataset")
    
    # Extract features and labels
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    feature_columns = [col for col in df.columns if col not in non_numeric_columns and col != target_column]
    
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found")
    
    print(f"Using {len(feature_columns)} feature columns")
    print(f"Target column: {target_column}")
    
    features = df[feature_columns].values
    labels = df[target_column].values
    
    # Handle any NaN values
    if np.isnan(features).any() or np.isnan(labels).any():
        print("Removing samples with NaN values...")
        mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
        features = features[mask]
        labels = labels[mask]
        print(f"Remaining samples: {len(features)}")
    
    # Clip labels to reasonable range if needed
    if labels.min() < 3.0 or labels.max() > 17.0:
        print(f"Clipping labels from [{labels.min():.2f}, {labels.max():.2f}] to [3.0, 17.0]")
        labels = np.clip(labels, 3.0, 17.0)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels range: [{labels.min():.2f}, {labels.max():.2f}] eV")
    print(f"Labels mean: {labels.mean():.2f} eV")
    
    return features, labels


if __name__ == "__main__":
    debug_qm9_data()
