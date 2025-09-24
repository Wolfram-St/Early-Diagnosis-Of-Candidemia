#!/usr/bin/env python3
"""
Enhanced Image Preprocessing for Candida Infection Classification

This script provides comprehensive image enhancement techniques specifically 
designed for microscopy images of Candida infections:

1. Brightness/Contrast adjustment (fix lighting variations)
2. Gaussian Blur (smooth out noise)  
3. CLAHE (Contrast Limited Adaptive Histogram Equalization) (enhance local contrast)
4. Unsharp Masking (sharpen details like fungal structures)
"""

import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImageProcessor:
    """Enhanced image processor for microscopy images"""
    
    def __init__(self, config=None):
        """Initialize with default or custom configuration"""
        self.config = config or self.get_default_config()
        
    @staticmethod
    def get_default_config():
        """Default processing parameters optimized for microscopy images"""
        return {
            'brightness_contrast': {
                'auto_adjust': True,  # Automatically detect optimal parameters
                'alpha': 1.2,         # Contrast multiplier (1.0 = no change)
                'beta': 15,           # Brightness offset (-100 to 100)
                'clip_limit': 2.0     # Prevent over-enhancement
            },
            'gaussian_blur': {
                'enabled': True,
                'kernel_size': (3, 3),  # Must be odd numbers
                'sigma_x': 0.8,         # Standard deviation in X direction
                'sigma_y': 0.8          # Standard deviation in Y direction
            },
            'clahe': {
                'enabled': True,
                'clip_limit': 3.0,      # Higher = more contrast enhancement
                'tile_grid_size': (8, 8), # Size of neighborhood area
                'color_space': 'LAB'    # LAB, YUV, or HSV
            },
            'unsharp_mask': {
                'enabled': True,
                'kernel_size': (5, 5),  # Blur kernel size
                'sigma': 1.0,           # Gaussian blur standard deviation
                'amount': 1.5,          # Strength of sharpening
                'threshold': 0          # Only sharpen if difference > threshold
            }
        }
    
    def auto_adjust_brightness_contrast(self, image):
        """Automatically adjust brightness and contrast based on image statistics"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Target values for microscopy images
        target_mean = 128  # Mid-range brightness
        target_std = 50    # Good contrast
        
        # Calculate adjustment parameters
        alpha = target_std / (std_intensity + 1e-10)  # Contrast
        beta = target_mean - alpha * mean_intensity   # Brightness
        
        # Clamp values to reasonable ranges
        alpha = np.clip(alpha, 0.5, 3.0)
        beta = np.clip(beta, -50, 50)
        
        return alpha, beta
    
    def adjust_brightness_contrast(self, image):
        """Adjust brightness and contrast with optional auto-detection"""
        config = self.config['brightness_contrast']
        
        if config['auto_adjust']:
            alpha, beta = self.auto_adjust_brightness_contrast(image)
        else:
            alpha, beta = config['alpha'], config['beta']
        
        # Apply adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Optional: Prevent over-enhancement by clipping extreme values
        if config.get('clip_limit', 0) > 0:
            clip_val = int(255 * config['clip_limit'] / 100)
            adjusted = np.clip(adjusted, clip_val, 255 - clip_val)
        
        return adjusted
    
    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur for noise reduction"""
        config = self.config['gaussian_blur']
        
        if not config['enabled']:
            return image
        
        return cv2.GaussianBlur(
            image, 
            config['kernel_size'], 
            sigmaX=config['sigma_x'],
            sigmaY=config['sigma_y']
        )
    
    def apply_clahe(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        config = self.config['clahe']
        
        if not config['enabled']:
            return image
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=config['clip_limit'],
            tileGridSize=config['tile_grid_size']
        )
        
        # Apply CLAHE based on color space
        color_space = config['color_space'].upper()
        
        if color_space == 'LAB':
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_clahe = clahe.apply(l)
            
            # Merge and convert back
            lab_clahe = cv2.merge((l_clahe, a, b))
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
        elif color_space == 'YUV':
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            
            # Apply CLAHE to Y channel
            y_clahe = clahe.apply(y)
            
            # Merge and convert back
            yuv_clahe = cv2.merge((y_clahe, u, v))
            result = cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)
            
        elif color_space == 'HSV':
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Apply CLAHE to V channel
            v_clahe = clahe.apply(v)
            
            # Merge and convert back
            hsv_clahe = cv2.merge((h, s, v_clahe))
            result = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
            
        else:
            # Apply to each channel separately (fallback)
            channels = cv2.split(image)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            result = cv2.merge(clahe_channels)
        
        return result
    
    def apply_unsharp_mask(self, image):
        """Apply unsharp masking for detail enhancement"""
        config = self.config['unsharp_mask']
        
        if not config['enabled']:
            return image
        
        # Create Gaussian blur
        kernel_size = config['kernel_size']
        sigma = config['sigma']
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # Calculate unsharp mask
        amount = config['amount']
        threshold = config['threshold']
        
        # Create mask
        mask = cv2.subtract(image, blurred)
        
        # Apply threshold if specified
        if threshold > 0:
            _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply unsharp mask
        sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
        
        return sharpened
    
    def process_image(self, image, show_steps=False):
        """Apply all processing steps to an image"""
        steps = []
        current_image = image.copy()
        
        if show_steps:
            steps.append(('Original', current_image.copy()))
        
        # Step 1: Brightness/Contrast adjustment
        current_image = self.adjust_brightness_contrast(current_image)
        if show_steps:
            steps.append(('Brightness/Contrast', current_image.copy()))
        
        # Step 2: Gaussian blur (noise reduction)
        current_image = self.apply_gaussian_blur(current_image)
        if show_steps:
            steps.append(('Gaussian Blur', current_image.copy()))
        
        # Step 3: CLAHE (local contrast enhancement)
        current_image = self.apply_clahe(current_image)
        if show_steps:
            steps.append(('CLAHE', current_image.copy()))
        
        # Step 4: Unsharp masking (detail enhancement)
        current_image = self.apply_unsharp_mask(current_image)
        if show_steps:
            steps.append(('Unsharp Mask', current_image.copy()))
        
        if show_steps:
            return current_image, steps
        else:
            return current_image
    
    def visualize_processing_steps(self, image, save_path=None):
        """Visualize all processing steps"""
        processed_image, steps = self.process_image(image, show_steps=True)
        
        # Create subplot layout
        n_steps = len(steps)
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (title, img) in enumerate(steps):
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(steps), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
        
        return processed_image

def process_dataset(input_dir, output_dir, config=None, show_progress=True, create_comparison=False):
    """Process entire dataset with enhanced preprocessing"""
    
    print("🧬 CANDIDA IMAGE PREPROCESSING")
    print("=" * 50)
    
    # Initialize processor
    processor = ImageProcessor(config)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    
    # Find all images
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Found {len(image_files)} images to process")
    print(f"🔧 Processing configuration:")
    
    # Print configuration
    for category, params in processor.config.items():
        print(f"  • {category.replace('_', ' ').title()}:")
        for key, value in params.items():
            if key != 'enabled' or value:
                print(f"    - {key}: {value}")
    
    print("-" * 50)
    
    # Process images
    processed_count = 0
    failed_count = 0
    
    progress_bar = tqdm(image_files, disable=not show_progress)
    for img_path in progress_bar:
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                failed_count += 1
                continue
            
            # Process image
            processed_image = processor.process_image(image)
            
            # Determine output path (preserve folder structure)
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save processed image
            cv2.imwrite(output_path, processed_image)
            processed_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'processed': processed_count,
                'failed': failed_count
            })
            
        except Exception as e:
            failed_count += 1
            print(f"❌ Failed to process {img_path}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Failed: {failed_count} images")
    print(f"📁 Output saved to: {output_dir}")
    
    # Create comparison visualization if requested
    if create_comparison and image_files:
        print("\n🖼️  Creating sample comparison...")
        sample_image_path = image_files[len(image_files)//2]  # Pick middle image
        sample_image = cv2.imread(sample_image_path)
        
        if sample_image is not None:
            comparison_path = os.path.join(output_dir, "processing_comparison.png")
            processor.visualize_processing_steps(sample_image, comparison_path)
    
    return processed_count, failed_count

def create_custom_config():
    """Create custom configuration for specific use cases"""
    configs = {
        'gentle': {
            'brightness_contrast': {'auto_adjust': True, 'alpha': 1.1, 'beta': 5},
            'gaussian_blur': {'enabled': True, 'kernel_size': (3, 3), 'sigma_x': 0.5, 'sigma_y': 0.5},
            'clahe': {'enabled': True, 'clip_limit': 2.0, 'tile_grid_size': (8, 8), 'color_space': 'LAB'},
            'unsharp_mask': {'enabled': True, 'kernel_size': (3, 3), 'sigma': 0.8, 'amount': 1.2, 'threshold': 0}
        },
        'aggressive': {
            'brightness_contrast': {'auto_adjust': True, 'alpha': 1.5, 'beta': 20},
            'gaussian_blur': {'enabled': True, 'kernel_size': (5, 5), 'sigma_x': 1.2, 'sigma_y': 1.2},
            'clahe': {'enabled': True, 'clip_limit': 4.0, 'tile_grid_size': (6, 6), 'color_space': 'LAB'},
            'unsharp_mask': {'enabled': True, 'kernel_size': (7, 7), 'sigma': 1.5, 'amount': 2.0, 'threshold': 5}
        },
        'detail_focused': {
            'brightness_contrast': {'auto_adjust': True, 'alpha': 1.3, 'beta': 10},
            'gaussian_blur': {'enabled': False},  # Skip blur to preserve details
            'clahe': {'enabled': True, 'clip_limit': 3.5, 'tile_grid_size': (12, 12), 'color_space': 'LAB'},
            'unsharp_mask': {'enabled': True, 'kernel_size': (5, 5), 'sigma': 1.0, 'amount': 2.5, 'threshold': 0}
        }
    }
    return configs

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Image Preprocessing for Candida Infection Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python image_process.py --input-dir dataset --output-dir processed_dataset
  
  # With visualization
  python image_process.py --input-dir dataset --output-dir processed_dataset --show-comparison
  
  # Use preset configuration
  python image_process.py --input-dir dataset --output-dir processed_dataset --preset aggressive
  
  # Process single image with visualization
  python image_process.py --input-dir single_image.tif --output-dir output --visualize
  
Processing Steps:
  1. Brightness/Contrast adjustment (fix lighting variations)
  2. Gaussian Blur (smooth out noise)
  3. CLAHE (enhance local contrast) 
  4. Unsharp Masking (sharpen fungal structures)
        """
    )
    
    parser.add_argument("--input-dir", required=True, 
                        help="Path to input images or directory")
    parser.add_argument("--output-dir", required=True,
                        help="Path to save processed images")
    parser.add_argument("--preset", choices=['gentle', 'aggressive', 'detail_focused'],
                        help="Use preset configuration")
    parser.add_argument("--config", 
                        help="Path to custom JSON configuration file")
    parser.add_argument("--show-comparison", action="store_true",
                        help="Create before/after comparison visualization")
    parser.add_argument("--visualize", action="store_true",
                        help="Show processing steps for sample image")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded custom configuration from {args.config}")
    elif args.preset:
        config = create_custom_config()[args.preset]
        print(f"✓ Using preset configuration: {args.preset}")
    
    # Check if input is single file or directory
    if os.path.isfile(args.input_dir):
        # Process single image
        print("🖼️  Processing single image...")
        
        image = cv2.imread(args.input_dir)
        if image is None:
            print(f"❌ Cannot read image: {args.input_dir}")
            return
        
        processor = ImageProcessor(config)
        
        if args.visualize:
            # Show processing steps
            os.makedirs(args.output_dir, exist_ok=True)
            visualization_path = os.path.join(args.output_dir, "processing_steps.png")
            processed_image = processor.visualize_processing_steps(image, visualization_path)
        else:
            processed_image = processor.process_image(image)
        
        # Save processed image
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.basename(args.input_dir)
        output_path = os.path.join(args.output_dir, f"processed_{output_filename}")
        cv2.imwrite(output_path, processed_image)
        print(f"✓ Processed image saved: {output_path}")
        
    else:
        # Process directory
        processed_count, failed_count = process_dataset(
            args.input_dir, 
            args.output_dir, 
            config=config,
            show_progress=not args.no_progress,
            create_comparison=args.show_comparison
        )
        
        if processed_count > 0:
            print(f"\n🎉 Successfully processed {processed_count} images!")
            if args.visualize and processed_count > 0:
                print("💡 Tip: Use --show-comparison to see before/after examples")
        else:
            print("\n❌ No images were processed successfully")

if __name__ == "__main__":
    main()
