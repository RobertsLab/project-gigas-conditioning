# Oyster Count Automation

A machine learning pipeline for automatically counting oysters in photographs using computer vision techniques.

## Overview

This system uses adaptive thresholding, morphological operations, and contour detection to identify and count individual oysters in high-resolution images. The algorithm has been optimized on a set of 8 labeled images containing between 99-116 oysters each.

## Directory Structure

```
count-automation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── oyster_counter.py           # Main counting algorithm
├── optimize_parameters.py      # Parameter optimization script
└── batch_process.py            # Batch processing utility (optional)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python-headless >= 4.8.0
- numpy >= 1.24.0
- scikit-image >= 0.21.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- pillow >= 10.0.0

## Usage

### Basic Usage - Count Oysters in a Single Image

```bash
python oyster_counter.py --input /path/to/image.jpeg --output /path/to/output_dir
```

### Evaluate on a Dataset of Labeled Images

```bash
python oyster_counter.py --input /path/to/image_directory --evaluate --output /path/to/output_dir
```

The filenames should follow the convention: `prefix-COUNT.jpeg` where COUNT is the actual number of oysters.

### Using Custom Parameters

```bash
python oyster_counter.py --input /path/to/image.jpeg --params custom_params.json
```

### Parameter Optimization

To optimize parameters on your labeled dataset:

```bash
python optimize_parameters.py        # Quick optimization (96 combinations)
python optimize_parameters.py --full # Full optimization (1728 combinations)
```

## Algorithm Description

### Pipeline Overview

1. **Image Preprocessing**
   - Resize image by 0.25x for faster processing (configurable)
   - Convert to grayscale
   - Apply Gaussian blur to reduce noise

2. **Oyster Detection**
   - Adaptive thresholding to handle varying lighting conditions
   - Morphological opening to remove small noise
   - Morphological closing to fill small holes
   - Contour detection to identify potential oysters

3. **Filtering and Counting**
   - Filter contours by area (min/max thresholds)
   - Calculate circularity: 4π × area / perimeter²
   - Filter by circularity threshold
   - Check aspect ratio (width/height)
   - Count valid contours as oysters

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resize_factor` | 0.25 | Scale factor for image resizing |
| `blur_kernel` | 5 | Gaussian blur kernel size |
| `adaptive_block_size` | 75* | Block size for adaptive thresholding |
| `adaptive_c` | 8* | Constant subtracted in adaptive threshold |
| `morph_kernel_size` | 7* | Morphological operation kernel size |
| `min_area` | 300* | Minimum contour area (pixels) |
| `max_area` | 40000* | Maximum contour area (pixels) |
| `circularity_threshold` | 0.25* | Minimum circularity (0-1) |
| `aspect_ratio_max` | 3.0 | Maximum width/height ratio |

*Optimized values based on current dataset

## Performance Metrics

### Current Results (8 images, 99-116 oysters each)

- **Mean Absolute Error**: 40.6 oysters
- **Mean Percentage Error**: 38.5%

### Detailed Results by Image

| Image | Ground Truth | Predicted | Error | % Error |
|-------|-------------|-----------|-------|---------|
| juvenile-102.jpeg | 102 | 72 | -30 | 29.4% |
| juvenile-103.jpeg | 103 | 53 | -50 | 48.5% |
| juvenile-106.jpeg | 106 | 60 | -46 | 43.4% |
| juvenile-114.jpeg | 114 | 76 | -38 | 33.3% |
| juvenile-116.jpeg | 116 | 71 | -45 | 38.8% |
| juvenile-99.jpeg | 99 | 66 | -33 | 33.3% |
| juvenile19-106.jpeg | 106 | 67 | -39 | 36.8% |
| juvenile20-99.jpeg | 99 | 55 | -44 | 44.4% |

### Analysis

The algorithm currently shows:
- Consistent undercounting (negative errors)
- Better performance on some images (29-36% error)
- More challenging cases with higher density (44-48% error)
- Average detection rate: ~61.5% of actual oysters

## Improvement Recommendations

### Short-term Improvements (Current Dataset)

1. **Fine-tune Detection Parameters**
   - Further optimize `min_area` and `max_area` thresholds
   - Adjust `circularity_threshold` for better shape matching
   - Experiment with different morphological kernel sizes

2. **Enhanced Preprocessing**
   - Test different color spaces (HSV, LAB) for better segmentation
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Experiment with bilateral filtering for edge-preserving smoothing

3. **Advanced Detection Methods**
   - Implement watershed segmentation for touching oysters
   - Use distance transform to separate clustered objects
   - Apply connected component analysis with stricter criteria

### Long-term Improvements (More Data)

1. **Expand Training Dataset**
   - **Priority**: Add 20-30 more images with known counts
   - Include images with varying conditions:
     - Different lighting (bright sun, overcast, shadows)
     - Different oyster densities (sparse: 20-50, medium: 50-100, dense: 100+)
     - Different backgrounds and surfaces
     - Various oyster sizes (juvenile vs. adult)
     - Different camera angles (overhead, angled)

2. **Implement Deep Learning**
   - With 30+ labeled images, train a custom object detection model:
     - YOLO (You Only Look Once) - Fast, real-time detection
     - Faster R-CNN - Higher accuracy, slower processing
     - EfficientDet - Balance of speed and accuracy
   - Use transfer learning from pre-trained models
   - Implement data augmentation (rotation, flip, brightness, crop)

3. **Develop Active Learning Pipeline**
   - Automatically identify images where model is uncertain
   - Prioritize these images for manual labeling
   - Incrementally retrain model with new labels

4. **Create Annotation Tool**
   - Build simple web interface for labeling oysters
   - Export labels in COCO or Pascal VOC format
   - Track inter-annotator agreement

5. **Ensemble Methods**
   - Combine multiple detection algorithms
   - Use voting or weighted averaging for final count
   - Improve robustness across different conditions

### Recommended Data Collection Strategy

**Phase 1** (10 additional images):
- 5 images with 50-80 oysters (medium density)
- 5 images with 120-150 oysters (high density)

**Phase 2** (10 additional images):
- 5 images in different lighting conditions
- 5 images with different backgrounds/surfaces

**Phase 3** (10+ images):
- Edge cases: very sparse (<30), very dense (>150)
- Different oyster life stages if applicable
- Various environmental conditions

### Validation Strategy

- Split data into training (70%), validation (15%), test (15%)
- Use cross-validation for small datasets
- Track metrics over time:
  - Mean Absolute Error (MAE)
  - Mean Percentage Error (MPE)
  - R² correlation between predicted and actual counts
  - Precision and recall at different thresholds

## Output Files

When running evaluation mode, the system generates:

1. **Visualization Images** (`detected_*.jpeg`)
   - Original image with detected oysters outlined in green
   - Bounding boxes around each detection
   - Count displayed on image

2. **Evaluation Results** (`evaluation_results.json`)
   - Detailed metrics for each image
   - Summary statistics
   - Individual image errors and predictions

3. **Optimization Results** (`optimization_results.json`)
   - All parameter combinations tested
   - Performance metrics for each
   - Best parameters identified

4. **Best Parameters** (`best_params.json`)
   - Optimal parameter values
   - Ready to use with `--params` flag

## Troubleshooting

### Common Issues

**Issue**: Severe undercounting
- **Solution**: Decrease `min_area` or increase `max_area`
- **Solution**: Decrease `circularity_threshold`
- **Solution**: Increase `adaptive_block_size`

**Issue**: Overcounting (many false positives)
- **Solution**: Increase `min_area` or decrease `max_area`
- **Solution**: Increase `circularity_threshold`
- **Solution**: Increase `morph_kernel_size`

**Issue**: Poor performance in bright/dark areas
- **Solution**: Adjust `adaptive_c` parameter
- **Solution**: Try different preprocessing techniques
- **Solution**: Apply histogram equalization

**Issue**: Touching oysters counted as one
- **Solution**: Implement watershed segmentation
- **Solution**: Decrease `morph_kernel_size` to reduce merging
- **Solution**: Use distance transform method

## Technical Notes

### Computational Performance

- Processing time: ~0.5-1 second per image (at 0.25x scale)
- Memory usage: ~200-500 MB depending on image size
- Scalable to batch processing

### Limitations

1. **Current Approach**
   - Works best with well-separated oysters
   - Struggles with heavily overlapping specimens
   - Sensitive to lighting variations
   - Limited by single-image training

2. **Image Quality Requirements**
   - Minimum resolution: 1000x1000 pixels
   - Clear focus (not blurry)
   - Reasonable contrast between oysters and background

3. **Biological Variations**
   - Assumes similar oyster sizes
   - May miscount shells or debris
   - Different life stages may need different parameters

## Contributing

To improve this system:

1. Add more labeled training images
2. Experiment with different algorithms
3. Optimize parameters for your specific use case
4. Report issues and results back to the team

## References

### Computer Vision Techniques
- Adaptive Thresholding: Automatically adjusts threshold based on local regions
- Morphological Operations: Clean up binary images (opening, closing, erosion, dilation)
- Contour Detection: Find boundaries of objects in binary images
- Watershed Segmentation: Separate touching objects

### Potential Deep Learning Approaches
- YOLO: "You Only Look Once" - Real-time object detection
- Faster R-CNN: Region-based Convolutional Neural Networks
- U-Net: Semantic segmentation architecture
- Transfer Learning: Use pre-trained models (ImageNet, COCO)

---

## Changelog

### 2025-10-20 16:36 UTC - Initial Development

**Created by**: GitHub Copilot  
**Status**: Initial implementation complete

#### Changes Made

1. **Created Directory Structure**
   - Set up `code/count-automation/` directory
   - Created README.md with comprehensive documentation

2. **Implemented Core Algorithm** (`oyster_counter.py`)
   - Developed OysterCounter class with configurable parameters
   - Implemented preprocessing pipeline:
     - Image resizing for performance
     - Grayscale conversion
     - Gaussian blur for noise reduction
   - Implemented detection pipeline:
     - Adaptive thresholding for varying lighting
     - Morphological operations (opening/closing)
     - Contour detection and filtering
   - Added shape filtering (area, circularity, aspect ratio)
   - Created visualization generation
   - Built command-line interface with argparse
   - Implemented evaluation mode for labeled datasets
   - Added filename parsing for ground truth extraction

3. **Implemented Parameter Optimization** (`optimize_parameters.py`)
   - Built grid search optimization framework
   - Defined parameter search space
   - Created quick and full optimization modes
   - Implemented results tracking and JSON export

4. **Created Requirements File** (`requirements.txt`)
   - Listed all Python dependencies
   - Specified minimum version requirements

5. **Conducted Initial Testing**
   - Ran baseline evaluation: MAE=67.0, MPE=63.4%
   - Optimized parameters across 96 combinations
   - Achieved improved results: MAE=40.6, MPE=38.5%
   - Generated visualization images for all 8 test images

6. **Documentation**
   - Created comprehensive README with:
     - Installation instructions
     - Usage examples
     - Algorithm description
     - Parameter documentation
     - Performance metrics
     - Improvement recommendations
     - Data collection strategy
     - Troubleshooting guide

#### Performance Summary

**Dataset**: 8 images (juvenile-99 to juvenile-116)
- Ground truth counts: 99-116 oysters per image
- Image resolution: 4032x3024 pixels (iPhone 14 Pro)

**Results**:
- Mean Absolute Error: 40.6 oysters
- Mean Percentage Error: 38.5%
- Detection rate: ~61.5% of actual oysters
- Consistent undercounting across all images

**Best Parameters**:
```json
{
  "resize_factor": 0.25,
  "blur_kernel": 5,
  "adaptive_block_size": 75,
  "adaptive_c": 8,
  "morph_kernel_size": 7,
  "min_area": 300,
  "max_area": 40000,
  "circularity_threshold": 0.25,
  "aspect_ratio_max": 3.0
}
```

#### Next Steps

1. Collect 20-30 additional labeled images with varying conditions
2. Implement advanced segmentation (watershed, distance transform)
3. Explore deep learning approaches with expanded dataset
4. Develop active learning pipeline for efficient labeling
5. Create ensemble methods for improved robustness

---

### Future Entries

Future updates to this codebase should be documented here with:
- Date and time (UTC)
- Author/contributor
- Description of changes
- Performance impact (if applicable)
- Any breaking changes or new dependencies
