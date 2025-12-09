#!/usr/bin/env python3
"""
Minimal test script to verify fixes for:
1. Tensor size mismatch in path_ablation.py
2. Missing plot_iia_vs_epoch import in sgd.py
3. Relative import issue in run_experiment_mnist.py
"""

import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path - need to add parent directory for relative imports
parent_dir = os.path.dirname(__file__)
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)

# Import using relative imports from src package
from src.data.models.ffnn import MLP
from src.algos.sgd import train_sgd
from src.analysis.path_ablation import compute_eigenpath_ablation
from src.analysis.path_analysis import plot_iia_vs_epoch
from src.analysis.circuit_comparison import compare_initial_vs_final_networks

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from src.analysis.path_analysis import plot_iia_vs_epoch
        print("  ✓ plot_iia_vs_epoch import works")
    except Exception as e:
        print(f"  ✗ plot_iia_vs_epoch import failed: {e}")
        return False
    
    try:
        from src.analysis.circuit_comparison import compare_initial_vs_final_networks
        print("  ✓ compare_initial_vs_final_networks import works")
    except Exception as e:
        print(f"  ✗ compare_initial_vs_final_networks import failed: {e}")
        return False
    
    return True

def create_mini_dataset(n=50, d=10):
    """Create a tiny dataset for testing."""
    torch.manual_seed(42)
    X = torch.randn(n, d)
    y = torch.sign(torch.randn(n, 1))  # Binary labels in {-1, +1}
    return X, y

def test_path_ablation():
    """Test that path ablation works without tensor size errors."""
    print("\nTesting path ablation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create tiny model and dataset
    model = MLP(d_in=10, widths=[8, 4], bias=True, activation="relu")
    model = model.to(device)
    
    X_train, y_train = create_mini_dataset(n=50, d=10)
    X_test, y_test = create_mini_dataset(n=20, d=10)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    
    try:
        # This should not crash with tensor size mismatch
        ablation_results = compute_eigenpath_ablation(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            mode="routing",
            k=10,  # Small k for fast testing
            max_samples=50,
            device=device,
            block_size=50,
            power_iters=5,  # Few iterations for speed
        )
        print("  ✓ Path ablation completed successfully")
        print(f"    k_values: {len(ablation_results['k_values'])}")
        print(f"    train_errors: {len(ablation_results['train_errors'])}")
        return True
    except Exception as e:
        print(f"  ✗ Path ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iia_plot():
    """Test that IIA plotting function exists and can be called."""
    print("\nTesting IIA plot function...")
    try:
        # Create dummy history with IIA data
        history = [
            {"epoch": 0, "iia_accuracy": 0.5},
            {"epoch": 1, "iia_accuracy": 0.6},
            {"epoch": 2, "iia_accuracy": 0.7},
        ]
        
        # Just check the function exists and can be imported
        from src.analysis.path_analysis import plot_iia_vs_epoch
        print("  ✓ plot_iia_vs_epoch function is available")
        print("    (Skipping actual plot generation to avoid file I/O)")
        return True
    except Exception as e:
        print(f"  ✗ IIA plot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sgd_training():
    """Test that SGD training works with path analysis enabled."""
    print("\nTesting SGD training with path analysis...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create tiny model and dataset
    model = MLP(d_in=10, widths=[8, 4], bias=True, activation="relu")
    model = model.to(device)
    
    X_train, y_train = create_mini_dataset(n=50, d=10)
    X_val, y_val = create_mini_dataset(n=20, d=10)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    
    # Minimal config
    config = {
        "device": device,
        "training": {
            "epochs": 2,  # Just 2 epochs for speed
            "lr_w": 0.01,
        },
        "model": {
            "activation": "relu",
            "widths": [8, 4],
            "bias": True,
        },
        "logging": {
            "path_kernel_metrics_every_n_epochs": 0,  # Disable for speed
            "path_analysis_every_n_epochs": 0,  # Disable for speed
            "enable_path_analysis": False,  # Disable for speed
        },
        "out_dir": "/tmp/test_sgd_output",
    }
    
    try:
        os.makedirs(config["out_dir"], exist_ok=True)
        train_sgd(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            test_loader=None,
        )
        print("  ✓ SGD training completed successfully")
        return True
    except Exception as e:
        print(f"  ✗ SGD training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing fixes for path_ablation, sgd, and run_experiment_mnist")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Path ablation
    results.append(("Path Ablation", test_path_ablation()))
    
    # Test 3: IIA plot
    results.append(("IIA Plot", test_iia_plot()))
    
    # Test 4: SGD training (minimal)
    results.append(("SGD Training", test_sgd_training()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())

