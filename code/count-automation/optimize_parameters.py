#!/usr/bin/env python3
"""
Parameter Optimization for Oyster Counter

This script performs grid search to find optimal parameters for the oyster counting algorithm.

Author: GitHub Copilot
Created: 2025-10-20
"""

import json
import itertools
from pathlib import Path
import numpy as np
from oyster_counter import OysterCounter, evaluate_on_dataset
import sys


def grid_search(image_dir: str, param_grid: dict, output_file: str = None):
    """
    Perform grid search over parameter combinations.
    
    Args:
        image_dir: Directory containing labeled images
        param_grid: Dictionary of parameter names to lists of values to try
        output_file: Optional file to save results
        
    Returns:
        Dictionary with best parameters and results
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    print("="*80)
    
    best_mae = float('inf')
    best_params = None
    best_results = None
    all_trials = []
    
    for i, combo in enumerate(combinations):
        # Create parameter dict
        params = OysterCounter._default_params()
        for name, value in zip(param_names, combo):
            params[name] = value
        
        # Test this combination
        counter = OysterCounter(params)
        results = evaluate_on_dataset(counter, image_dir, output_dir=None)
        
        if results['num_images'] > 0:
            mae = results['mean_absolute_error']
            mpe = results['mean_percentage_error']
            
            trial_info = {
                'trial': i + 1,
                'params': params,
                'mae': mae,
                'mpe': mpe
            }
            all_trials.append(trial_info)
            
            print(f"\nTrial {i+1}/{len(combinations)}")
            print(f"Parameters: {combo}")
            print(f"MAE: {mae:.2f}, MPE: {mpe:.1f}%")
            
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                best_results = results
                print("*** New best result! ***")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest Mean Absolute Error: {best_mae:.2f}")
    print(f"Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    output = {
        'best_params': best_params,
        'best_mae': best_mae,
        'best_mpe': best_results['mean_percentage_error'] if best_results else None,
        'all_trials': all_trials
    }
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nOptimization results saved to: {output_file}")
    
    return output


def main():
    """Main function."""
    # Define parameter grid for optimization
    # These ranges are based on typical values for blob/object detection
    param_grid = {
        'adaptive_block_size': [51, 75, 101, 151],
        'adaptive_c': [5, 10, 15, 20],
        'min_area': [200, 400, 600, 800],
        'max_area': [30000, 50000, 70000],
        'circularity_threshold': [0.2, 0.3, 0.4],
        'morph_kernel_size': [3, 5, 7],
    }
    
    # Use a smaller grid for faster optimization
    # You can expand this for more thorough optimization
    quick_param_grid = {
        'adaptive_block_size': [75, 101],
        'adaptive_c': [8, 12],
        'min_area': [300, 500, 700],
        'max_area': [40000, 60000],
        'circularity_threshold': [0.25, 0.35],
        'morph_kernel_size': [5, 7],
    }
    
    image_dir = '../../data/images'
    output_file = '../../output/count-automation/optimization_results.json'
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting parameter optimization...")
    print(f"Using {'quick' if len(sys.argv) == 1 else 'full'} parameter grid")
    print()
    
    # Run optimization with quick grid by default
    # Use full grid if --full argument is provided
    use_full = len(sys.argv) > 1 and sys.argv[1] == '--full'
    grid = param_grid if use_full else quick_param_grid
    
    results = grid_search(image_dir, grid, output_file)
    
    # Save best parameters to a separate file for easy use
    best_params_file = '../../output/count-automation/best_params.json'
    with open(best_params_file, 'w') as f:
        json.dump(results['best_params'], f, indent=2)
    print(f"Best parameters saved to: {best_params_file}")


if __name__ == '__main__':
    main()
