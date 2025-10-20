#!/usr/bin/env python3
"""
Example Usage of Oyster Counter

This script demonstrates how to use the OysterCounter class programmatically.

Author: GitHub Copilot
Created: 2025-10-20
"""

from oyster_counter import OysterCounter
import json
from pathlib import Path


def example_1_single_image():
    """Example 1: Count oysters in a single image with default parameters."""
    print("="*60)
    print("Example 1: Single Image with Default Parameters")
    print("="*60)
    
    # Create counter with default parameters
    counter = OysterCounter()
    
    # Count oysters in an image
    image_path = "../../data/images/juvenile-102.jpeg"
    count, visualization = counter.count_oysters(image_path)
    
    print(f"Image: {image_path}")
    print(f"Detected oysters: {count}")
    print()


def example_2_custom_parameters():
    """Example 2: Count oysters using custom parameters."""
    print("="*60)
    print("Example 2: Single Image with Custom Parameters")
    print("="*60)
    
    # Define custom parameters
    custom_params = {
        'resize_factor': 0.25,
        'blur_kernel': 5,
        'adaptive_block_size': 75,
        'adaptive_c': 8,
        'morph_kernel_size': 7,
        'min_area': 300,
        'max_area': 40000,
        'circularity_threshold': 0.25,
        'aspect_ratio_max': 3.0,
    }
    
    # Create counter with custom parameters
    counter = OysterCounter(params=custom_params)
    
    # Count oysters
    image_path = "../../data/images/juvenile-114.jpeg"
    count, visualization = counter.count_oysters(image_path)
    
    print(f"Image: {image_path}")
    print(f"Detected oysters: {count}")
    print()


def example_3_load_parameters_from_file():
    """Example 3: Load optimized parameters from JSON file."""
    print("="*60)
    print("Example 3: Using Optimized Parameters from File")
    print("="*60)
    
    # Load parameters from file
    params_file = "../../output/count-automation/best_params.json"
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Create counter with loaded parameters
    counter = OysterCounter(params=params)
    
    # Count oysters in multiple images
    image_dir = Path("../../data/images")
    images = sorted(image_dir.glob("*.jpeg"))[:3]  # First 3 images
    
    for img in images:
        count, _ = counter.count_oysters(str(img))
        print(f"{img.name}: {count} oysters")
    
    print()


def example_4_programmatic_evaluation():
    """Example 4: Evaluate performance programmatically."""
    print("="*60)
    print("Example 4: Programmatic Evaluation")
    print("="*60)
    
    from oyster_counter import evaluate_on_dataset
    
    # Load optimized parameters
    params_file = "../../output/count-automation/best_params.json"
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Create counter
    counter = OysterCounter(params=params)
    
    # Evaluate on dataset
    results = evaluate_on_dataset(
        counter,
        "../../data/images",
        output_dir=None  # Don't save visualizations
    )
    
    print(f"\nEvaluation Results:")
    print(f"Number of images: {results['num_images']}")
    print(f"Mean Absolute Error: {results['mean_absolute_error']:.2f}")
    print(f"Mean Percentage Error: {results['mean_percentage_error']:.1f}%")
    print()


def example_5_adjust_parameters():
    """Example 5: Experiment with different parameter values."""
    print("="*60)
    print("Example 5: Experimenting with Parameters")
    print("="*60)
    
    image_path = "../../data/images/juvenile-99.jpeg"
    
    # Test with different min_area thresholds
    min_areas = [200, 300, 400, 500]
    
    print(f"Testing different min_area values on {Path(image_path).name}:\n")
    
    for min_area in min_areas:
        # Start with default parameters
        params = OysterCounter._default_params()
        # Override specific parameters
        params['min_area'] = min_area
        params['adaptive_block_size'] = 75
        params['adaptive_c'] = 8
        params['morph_kernel_size'] = 7
        
        counter = OysterCounter(params=params)
        count, _ = counter.count_oysters(image_path)
        
        print(f"min_area={min_area:4d}: {count:3d} oysters detected")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("OYSTER COUNTER - USAGE EXAMPLES")
    print("="*60 + "\n")
    
    # Run examples
    example_1_single_image()
    example_2_custom_parameters()
    example_3_load_parameters_from_file()
    example_4_programmatic_evaluation()
    example_5_adjust_parameters()
    
    print("="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
