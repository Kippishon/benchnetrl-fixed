#!/usr/bin/env python3
"""
SAC Test Script for BenchNetRL
Tests SAC implementation on simple environments before full MuJoCo runs.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

# Test configuration
TEST_CONFIGS = [
    {
        "name": "Pendulum-v1 (Quick Validation)",
        "env": "Pendulum-v1",
        "timesteps": 50000,
        "expected_time": "~5 minutes",
        "expected_return": "-200 to -150 (Pendulum is negative reward)",
    },
    {
        "name": "Hopper-v4 (Short Run)",
        "env": "Hopper-v4",
        "timesteps": 100000,
        "expected_time": "~15 minutes",
        "expected_return": "500-1500 (improving trend)",
    },
    {
        "name": "HalfCheetah-v4 (Full Pilot)",
        "env": "HalfCheetah-v4", 
        "timesteps": 500000,
        "expected_time": "~60-90 minutes",
        "expected_return": "3000-8000 (SAC should reach 5000+)",
    },
]

# SAC hyperparameters for testing (conservative defaults)
SAC_HYPERPARAMS = {
    "learning-rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch-size": 256,
    "buffer-size": 1000000,
    "learning-starts": 5000,  # Reduced for testing
    "warmup-steps": 1000,
    "hidden-dim": 256,
    "policy-frequency": 2,
    "updates-per-step": 1,
    "autotune": True,
    "alpha": 0.2,
}


def run_sac_test(env_id, timesteps, seed=42, capture_video=False):
    """Run SAC on a single environment."""
    
    cmd = [
        sys.executable, "sac.py",
        "--gym-id", env_id,
        "--total-timesteps", str(timesteps),
        "--seed", str(seed),
        "--learning-rate", str(SAC_HYPERPARAMS["learning-rate"]),
        "--gamma", str(SAC_HYPERPARAMS["gamma"]),
        "--tau", str(SAC_HYPERPARAMS["tau"]),
        "--batch-size", str(SAC_HYPERPARAMS["batch-size"]),
        "--buffer-size", str(SAC_HYPERPARAMS["buffer-size"]),
        "--learning-starts", str(SAC_HYPERPARAMS["learning-starts"]),
        "--warmup-steps", str(SAC_HYPERPARAMS["warmup-steps"]),
        "--hidden-dim", str(SAC_HYPERPARAMS["hidden-dim"]),
        "--policy-frequency", str(SAC_HYPERPARAMS["policy-frequency"]),
        "--updates-per-step", str(SAC_HYPERPARAMS["updates-per-step"]),
        "--alpha", str(SAC_HYPERPARAMS["alpha"]),
        "--exp-name", f"sac_test_{env_id.lower().replace('-', '_')}",
    ]
    
    if SAC_HYPERPARAMS["autotune"]:
        cmd.append("--autotune")
    
    if capture_video:
        cmd.append("--capture-video")
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True,
            check=True,
        )
        elapsed = time.time() - start_time
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"ERROR: SAC test failed with exit code {e.returncode}")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: Unexpected error: {e}")
        return False, elapsed


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        import gymnasium
        import numpy as np
        import wandb
        print(f"  torch: {torch.__version__}")
        print(f"  gymnasium: {gymnasium.__version__}")
        print(f"  numpy: {np.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print("  All imports OK")
        return True
    except ImportError as e:
        print(f"  IMPORT ERROR: {e}")
        return False


def main():
    """Main test runner."""
    print("="*60)
    print("BenchNetRL SAC Implementation Test")
    print("="*60)
    
    # Test 1: Import check
    if not test_imports():
        print("\nFAILED: Import check failed. Please install requirements.")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        selected_tests = [t for t in TEST_CONFIGS if t["name"].lower().startswith(test_name.lower())]
        if not selected_tests:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {[t['name'] for t in TEST_CONFIGS]}")
            sys.exit(1)
    else:
        # Default: quick test only
        selected_tests = [TEST_CONFIGS[0]]
        print("\nRunning quick validation test (Pendulum-v1)")
        print("For full tests, run: python test_sac.py <test_name>")
        print(f"Available tests: {[t['name'] for t in TEST_CONFIGS]}")
    
    # Run selected tests
    results = []
    for test in selected_tests:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"Environment: {test['env']}")
        print(f"Timesteps: {test['timesteps']}")
        print(f"Expected time: {test['expected_time']}")
        print(f"Expected return: {test['expected_return']}")
        print(f"{'='*60}")
        
        success, elapsed = run_sac_test(
            test['env'],
            test['timesteps'],
        )
        
        results.append({
            'name': test['name'],
            'success': success,
            'elapsed': elapsed,
        })
        
        if not success:
            print(f"\nTEST FAILED: {test['name']}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for r in results:
        status = "PASSED" if r['success'] else "FAILED"
        print(f"  {r['name']}: {status} ({r['elapsed']:.1f}s)")
    
    all_passed = all(r['success'] for r in results)
    
    if all_passed:
        print("\nAll tests passed! SAC implementation is working correctly.")
        print("\nNext steps:")
        print("  1. Run full training on MuJoCo environments")
        print("  2. Compare results with PPO baselines")
        print("  3. Proceed to multi-layer architecture experiments")
    else:
        print("\nSome tests failed. Please check the error output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
