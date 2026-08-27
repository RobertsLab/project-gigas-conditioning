#!/usr/bin/env python3
"""
Batch Processing Utility for Oyster Counter

Process multiple images and generate a summary report.

Author: GitHub Copilot
Created: 2025-10-20
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from oyster_counter import OysterCounter
import cv2
import sys


def batch_process(input_dir: str, output_dir: str, params_file: str = None,
                 generate_vis: bool = True) -> pd.DataFrame:
    """
    Process all images in a directory and generate summary report.
    
    Args:
        input_dir: Directory containing images to process
        output_dir: Directory to save results
        params_file: Optional JSON file with custom parameters
        generate_vis: Whether to generate visualization images
        
    Returns:
        DataFrame with results for all images
    """
    # Load parameters if provided
    params = None
    if params_file:
        with open(params_file, 'r') as f:
            params = json.load(f)
    
    # Create counter
    counter = OysterCounter(params)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    input_path = Path(input_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(image_files)} images to process")
    print("="*60)
    
    results = []
    
    for i, img_file in enumerate(sorted(image_files)):
        try:
            print(f"\nProcessing {i+1}/{len(image_files)}: {img_file.name}")
            
            # Count oysters
            count, vis_image = counter.count_oysters(str(img_file))
            
            print(f"  Detected: {count} oysters")
            
            results.append({
                'filename': img_file.name,
                'count': count,
                'image_path': str(img_file.absolute())
            })
            
            # Save visualization if requested
            if generate_vis:
                vis_file = output_path / f"detected_{img_file.name}"
                cv2.imwrite(str(vis_file), vis_image)
                print(f"  Saved visualization: {vis_file.name}")
        
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
            results.append({
                'filename': img_file.name,
                'count': -1,
                'error': str(e),
                'image_path': str(img_file.absolute())
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_file = output_path / 'batch_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_file}")
    
    # Save to JSON
    json_file = output_path / 'batch_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_file}")
    
    # Print summary statistics
    if len(df[df['count'] >= 0]) > 0:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total images processed: {len(df)}")
        print(f"Successful: {len(df[df['count'] >= 0])}")
        print(f"Failed: {len(df[df['count'] < 0])}")
        
        valid_counts = df[df['count'] >= 0]['count']
        if len(valid_counts) > 0:
            print(f"\nOyster Count Statistics:")
            print(f"  Mean: {valid_counts.mean():.1f}")
            print(f"  Median: {valid_counts.median():.1f}")
            print(f"  Min: {valid_counts.min()}")
            print(f"  Max: {valid_counts.max()}")
            print(f"  Std Dev: {valid_counts.std():.1f}")
    
    return df


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Batch process images for oyster counting'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for results')
    parser.add_argument('--params', '-p',
                       help='JSON file with custom parameters')
    parser.add_argument('--no-vis', action='store_true',
                       help='Skip generating visualization images')
    
    args = parser.parse_args()
    
    print("Oyster Counter - Batch Processing")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.params:
        print(f"Parameters file: {args.params}")
    print()
    
    # Process images
    df = batch_process(
        args.input,
        args.output,
        args.params,
        generate_vis=not args.no_vis
    )
    
    if len(df) == 0:
        print("No images were processed successfully.")
        sys.exit(1)
    
    print("\nBatch processing complete!")


if __name__ == '__main__':
    main()
