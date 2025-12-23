#!/usr/bin/env python3
"""
Oyster Counter - Machine Learning Pipeline for Counting Oysters in Images

This script uses computer vision techniques to detect and count oysters in photographs.
It employs adaptive thresholding, morphological operations, and contour detection.

Author: GitHub Copilot
Created: 2025-10-20
"""

import cv2
import numpy as np
from skimage import morphology, measure
from pathlib import Path
import json
import argparse
from typing import Tuple, Dict, List
import sys


class OysterCounter:
    """
    A class for detecting and counting oysters in images using computer vision.
    
    Attributes:
        params (dict): Dictionary of tunable parameters for the detection algorithm
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize the OysterCounter with detection parameters.
        
        Args:
            params: Dictionary of parameters. If None, uses default values.
        """
        self.params = params or self._default_params()
    
    @staticmethod
    def _default_params() -> Dict:
        """Return default detection parameters (internal use)."""
        return {
            'resize_factor': 0.25,  # Resize images for faster processing
            'blur_kernel': 5,  # Gaussian blur kernel size
            'adaptive_block_size': 101,  # Block size for adaptive threshold
            'adaptive_c': 10,  # Constant subtracted from mean in adaptive threshold
            'morph_kernel_size': 5,  # Morphological operation kernel size
            'min_area': 500,  # Minimum contour area (in resized image)
            'max_area': 50000,  # Maximum contour area (in resized image)
            'circularity_threshold': 0.3,  # Minimum circularity (0-1)
            'aspect_ratio_max': 3.0,  # Maximum aspect ratio
        }
    
    @staticmethod
    def get_default_params() -> Dict:
        """
        Get a copy of the default detection parameters.
        
        Returns:
            Dictionary containing default parameter values
        """
        return OysterCounter._default_params().copy()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for oyster detection.
        
        Args:
            image: Input BGR image from OpenCV
            
        Returns:
            Preprocessed grayscale image
        """
        # Resize for faster processing
        scale = self.params['resize_factor']
        resized = cv2.resize(image, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, 
                                   (self.params['blur_kernel'], 
                                    self.params['blur_kernel']), 0)
        
        return blurred
    
    def detect_oysters(self, preprocessed: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect oysters in preprocessed image using contour detection.
        
        Args:
            preprocessed: Preprocessed grayscale image
            
        Returns:
            Tuple of (list of contours, binary mask)
        """
        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            preprocessed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.params['adaptive_block_size'],
            self.params['adaptive_c']
        )
        
        # Morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.params['morph_kernel_size'], self.params['morph_kernel_size'])
        )
        
        # Opening to remove small noise
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Closing to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.params['min_area'] or area > self.params['max_area']:
                continue
            
            # Calculate shape features
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity: 4π*area/perimeter²
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.params['circularity_threshold']:
                continue
            
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            if aspect_ratio > self.params['aspect_ratio_max']:
                continue
            
            valid_contours.append(contour)
        
        return valid_contours, closed
    
    def count_oysters(self, image_path: str) -> Tuple[int, np.ndarray]:
        """
        Count oysters in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (count, visualization image)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Detect oysters
        contours, binary_mask = self.detect_oysters(preprocessed)
        
        # Create visualization
        vis = self.visualize_results(image, contours)
        
        return len(contours), vis
    
    def visualize_results(self, original_image: np.ndarray, 
                         contours: List, show_contours: bool = True) -> np.ndarray:
        """
        Create a visualization of detection results.
        
        Args:
            original_image: Original BGR image
            contours: List of detected contours (in resized coordinates)
            show_contours: Whether to draw contours on the image
            
        Returns:
            Visualization image
        """
        # Create a copy for visualization
        scale = self.params['resize_factor']
        vis = cv2.resize(original_image, None, fx=scale, fy=scale, 
                        interpolation=cv2.INTER_AREA)
        
        if show_contours and len(contours) > 0:
            # Draw contours in green
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            
            # Draw bounding boxes and labels
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, str(i + 1), (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add count text
        count_text = f"Count: {len(contours)}"
        cv2.putText(vis, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis


def parse_filename(filename: str) -> int:
    """
    Extract the oyster count from the filename.
    
    Args:
        filename: Name of the image file (e.g., 'juvenile-102.jpeg')
        
    Returns:
        The actual oyster count
    """
    # Remove extension and split by dash
    name = Path(filename).stem
    parts = name.split('-')
    
    # Get the last part which should be the count
    if len(parts) >= 2:
        return int(parts[-1])
    
    raise ValueError(f"Could not parse count from filename: {filename}")


def evaluate_on_dataset(counter: OysterCounter, image_dir: str, 
                       output_dir: str = None) -> Dict:
    """
    Evaluate the counter on a dataset of labeled images.
    
    Args:
        counter: OysterCounter instance
        image_dir: Directory containing labeled images
        output_dir: Optional directory to save visualization images
        
    Returns:
        Dictionary with evaluation metrics
    """
    image_path = Path(image_dir)
    images = sorted(image_path.glob("*.jpeg")) + sorted(image_path.glob("*.jpg"))
    
    results = []
    
    for img_file in images:
        try:
            # Get ground truth from filename
            ground_truth = parse_filename(img_file.name)
            
            # Count oysters
            predicted_count, vis_image = counter.count_oysters(str(img_file))
            
            # Calculate error
            error = predicted_count - ground_truth
            abs_error = abs(error)
            pct_error = (abs_error / ground_truth * 100) if ground_truth > 0 else 0
            
            results.append({
                'filename': img_file.name,
                'ground_truth': ground_truth,
                'predicted': predicted_count,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error
            })
            
            print(f"{img_file.name}: GT={ground_truth}, Pred={predicted_count}, "
                  f"Error={error}, %Error={pct_error:.1f}%")
            
            # Save visualization if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"detected_{img_file.name}"
                cv2.imwrite(str(output_file), vis_image)
        
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    # Calculate summary statistics
    if results:
        total_abs_error = sum(r['abs_error'] for r in results)
        mean_abs_error = total_abs_error / len(results)
        mean_pct_error = sum(r['pct_error'] for r in results) / len(results)
        
        summary = {
            'num_images': len(results),
            'mean_absolute_error': mean_abs_error,
            'mean_percentage_error': mean_pct_error,
            'results': results
        }
    else:
        summary = {'num_images': 0, 'results': []}
    
    return summary


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Count oysters in images using computer vision'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', '-o', 
                       help='Output directory for visualizations')
    parser.add_argument('--params', '-p',
                       help='JSON file with custom parameters')
    parser.add_argument('--evaluate', '-e', action='store_true',
                       help='Evaluate on labeled dataset')
    
    args = parser.parse_args()
    
    # Load parameters if provided
    params = None
    if args.params:
        with open(args.params, 'r') as f:
            params = json.load(f)
    
    # Create counter
    counter = OysterCounter(params)
    
    input_path = Path(args.input)
    
    if args.evaluate:
        # Evaluate on dataset
        if not input_path.is_dir():
            print("Error: --evaluate requires a directory as input")
            sys.exit(1)
        
        print(f"Evaluating on images in: {input_path}")
        results = evaluate_on_dataset(counter, str(input_path), args.output)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Number of images: {results['num_images']}")
        if results['num_images'] > 0:
            print(f"Mean Absolute Error: {results['mean_absolute_error']:.2f}")
            print(f"Mean Percentage Error: {results['mean_percentage_error']:.1f}%")
        
        # Save results to JSON
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            results_file = output_path / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
    
    elif input_path.is_file():
        # Process single image
        count, vis = counter.count_oysters(str(input_path))
        print(f"Detected {count} oysters in {input_path.name}")
        
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"detected_{input_path.name}"
            cv2.imwrite(str(output_file), vis)
            print(f"Visualization saved to: {output_file}")
    
    else:
        print("Error: Input must be a file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
