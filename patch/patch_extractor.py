#!/usr/bin/env python3
"""
Enhanced Patch Extraction for Candida Infection Classification

This script extracts patches from images to create a larger training dataset,
which can improve model performance, especially for detecting fine-grained 
features in microscopy images.
"""

import os
import cv2
import numpy as np
import argparse
import json
from tqdm.auto import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PatchExtractor:
    """Enhanced patch extractor for microscopy images"""
    
    def __init__(self, patch_size=128, stride=64, min_patch_std=10, quality_threshold=0.3):
        """
        Initialize patch extractor
        
        Args:
            patch_size: Size of square patches (default: 128x128)
            stride: Step size between patches (smaller = more overlap)
            min_patch_std: Minimum standard deviation to filter out blank patches
            quality_threshold: Minimum quality score (0-1) to keep patches
        """
        self.patch_size = patch_size
        self.stride = stride
        self.min_patch_std = min_patch_std
        self.quality_threshold = quality_threshold
    
    def calculate_patch_quality(self, patch):
        """Calculate quality score for a patch (0-1, higher is better)"""
        # Convert to grayscale for analysis
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        
        # Calculate various quality metrics
        std_dev = np.std(gray)
        mean_intensity = np.mean(gray)
        
        # Edge detection to measure structure content
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (patch.shape[0] * patch.shape[1])
        
        # Calculate gradient magnitude (texture measure)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_mag)
        
        # Combine metrics into quality score (normalized 0-1)
        contrast_score = min(std_dev / 50.0, 1.0)  # Good contrast
        edge_score = min(edge_density * 10, 1.0)   # Structural content
        gradient_score = min(avg_gradient / 30.0, 1.0)  # Texture richness
        brightness_score = 1.0 - abs(mean_intensity - 128) / 128.0  # Good exposure
        
        # Weighted combination
        quality_score = (
            0.3 * contrast_score +
            0.3 * edge_score +
            0.2 * gradient_score +
            0.2 * brightness_score
        )
        
        return quality_score
    
    def extract_patches(self, image, image_name="", filter_quality=True):
        """
        Extract patches from an image
        
        Args:
            image: Input image (BGR format)
            image_name: Name for logging purposes
            filter_quality: Whether to filter patches by quality
            
        Returns:
            List of (patch, quality_score) tuples
        """
        if image is None:
            return []
        
        patches = []
        h, w = image.shape[:2]
        
        # Calculate number of patches
        n_patches_y = (h - self.patch_size) // self.stride + 1
        n_patches_x = (w - self.patch_size) // self.stride + 1
        total_patches = n_patches_y * n_patches_x
        
        patch_count = 0
        kept_count = 0
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_count += 1
                
                # Quality filtering
                if filter_quality:
                    quality_score = self.calculate_patch_quality(patch)
                    
                    # Skip low-quality patches
                    if quality_score < self.quality_threshold:
                        continue
                    
                    # Skip patches with too little variation (likely background)
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
                    if np.std(gray_patch) < self.min_patch_std:
                        continue
                else:
                    quality_score = 1.0
                
                patches.append((patch, quality_score))
                kept_count += 1
        
        if image_name:
            retention_rate = (kept_count / patch_count) * 100 if patch_count > 0 else 0
            print(f"  📄 {image_name}: {kept_count}/{patch_count} patches kept ({retention_rate:.1f}%)")
        
        return patches
    
    def process_dataset(self, input_dir, output_dir, preserve_structure=True, show_progress=True):
        """
        Process entire dataset and extract patches
        
        Args:
            input_dir: Input directory with class subdirectories
            output_dir: Output directory for patches
            preserve_structure: Whether to maintain train/val structure
            show_progress: Show progress bars
            
        Returns:
            Dictionary with processing statistics
        """
        print("🧬 CANDIDA PATCH EXTRACTION")
        print("=" * 60)
        print(f"📁 Input: {input_dir}")
        print(f"📁 Output: {output_dir}")
        print(f"🔧 Patch size: {self.patch_size}x{self.patch_size}")
        print(f"📏 Stride: {self.stride}")
        print(f"🎯 Quality threshold: {self.quality_threshold}")
        print("-" * 60)
        
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics tracking
        stats = {
            'total_images': 0,
            'total_patches': 0,
            'patches_per_class': {},
            'images_per_class': {},
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        # Process each subset (train/val or direct classes)
        for root, dirs, files in os.walk(input_dir):
            # Skip empty directories
            image_files = [f for f in files if f.lower().endswith(image_extensions)]
            if not image_files:
                continue
            
            # Determine class name and output path
            rel_path = os.path.relpath(root, input_dir)
            
            if preserve_structure and rel_path != '.':
                # Maintain directory structure (e.g., train/ca, val/ca)
                output_class_dir = os.path.join(output_dir, rel_path)
                class_name = os.path.basename(rel_path)
                split_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) != '.' else 'root'
            else:
                # Direct class directories
                output_class_dir = os.path.join(output_dir, os.path.basename(root))
                class_name = os.path.basename(root)
                split_name = 'root'
            
            # Create output directory
            os.makedirs(output_class_dir, exist_ok=True)
            
            print(f"\n📂 Processing {rel_path} ({len(image_files)} images)")
            
            # Initialize class statistics
            if class_name not in stats['patches_per_class']:
                stats['patches_per_class'][class_name] = 0
                stats['images_per_class'][class_name] = 0
            
            # Process images in this directory
            if show_progress:
                progress_bar = tqdm(image_files, desc=f"Extracting patches", leave=False)
            else:
                progress_bar = image_files
            
            for img_file in progress_bar:
                img_path = os.path.join(root, img_file)
                
                try:
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"  ⚠️  Could not read: {img_file}")
                        continue
                    
                    # Extract patches
                    patches = self.extract_patches(image, img_file, filter_quality=True)
                    
                    if not patches:
                        print(f"  ⚠️  No quality patches from: {img_file}")
                        continue
                    
                    # Save patches
                    base_name = os.path.splitext(img_file)[0]
                    
                    for i, (patch, quality_score) in enumerate(patches):
                        patch_filename = f"{base_name}_patch_{i:03d}_q{quality_score:.2f}.png"
                        patch_path = os.path.join(output_class_dir, patch_filename)
                        
                        # Save patch
                        cv2.imwrite(patch_path, patch)
                    
                    # Update statistics
                    stats['total_images'] += 1
                    stats['total_patches'] += len(patches)
                    stats['patches_per_class'][class_name] += len(patches)
                    stats['images_per_class'][class_name] += 1
                    
                    # Update progress bar
                    if show_progress:
                        progress_bar.set_postfix({
                            'patches': len(patches),
                            'total': stats['total_patches']
                        })
                
                except Exception as e:
                    print(f"  ❌ Error processing {img_file}: {str(e)}")
        
        # Calculate processing time
        end_time = datetime.now()
        stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 PATCH EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"⏱️  Processing time: {stats['processing_time']:.1f} seconds")
        print(f"🖼️  Total images processed: {stats['total_images']}")
        print(f"🧩 Total patches extracted: {stats['total_patches']}")
        print(f"📈 Average patches per image: {stats['total_patches'] / max(stats['total_images'], 1):.1f}")
        
        print("\n📋 Patches per class:")
        for class_name, patch_count in stats['patches_per_class'].items():
            image_count = stats['images_per_class'][class_name]
            avg_patches = patch_count / max(image_count, 1)
            print(f"  • {class_name}: {patch_count} patches from {image_count} images (avg: {avg_patches:.1f})")
        
        # Save statistics
        stats_file = os.path.join(output_dir, "patch_extraction_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved: {stats_file}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description="Extract patches from Candida infection images for enhanced training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic patch extraction
  python patch_extractor.py --input-dir my_dataset --output-dir patched_dataset
  
  # Custom patch size and stride
  python patch_extractor.py --input-dir my_dataset --output-dir patched_dataset --patch-size 256 --stride 128
  
  # Higher quality filtering
  python patch_extractor.py --input-dir my_dataset --output-dir patched_dataset --quality-threshold 0.5
  
  # Process enhanced images
  python patch_extractor.py --input-dir enhanced_dataset --output-dir patched_enhanced_dataset

Patch extraction creates smaller image regions that can help models learn:
- Fine-grained fungal structures
- Local texture patterns  
- Detailed morphological features
- Better generalization with more training samples
        """
    )
    
    parser.add_argument("--input-dir", required=True,
                        help="Input directory with train/val or class subdirectories")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for extracted patches")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Size of square patches (default: 128)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Step size between patches - smaller = more overlap (default: 64)")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                        help="Minimum quality score to keep patches 0-1 (default: 0.3)")
    parser.add_argument("--min-std", type=float, default=10.0,
                        help="Minimum standard deviation to filter blank patches (default: 10.0)")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable quality filtering (keep all patches)")
    parser.add_argument("--no-structure", action="store_true",
                        help="Don't preserve train/val directory structure")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.patch_size <= 0 or args.stride <= 0:
        parser.error("Patch size and stride must be positive")
    
    if args.stride > args.patch_size:
        print("⚠️  Warning: Stride > patch_size will result in gaps between patches")
    
    if args.quality_threshold < 0 or args.quality_threshold > 1:
        parser.error("Quality threshold must be between 0 and 1")
    
    # Set quality parameters
    quality_threshold = 0.0 if args.no_quality_filter else args.quality_threshold
    
    # Create patch extractor
    extractor = PatchExtractor(
        patch_size=args.patch_size,
        stride=args.stride,
        min_patch_std=args.min_std,
        quality_threshold=quality_threshold
    )
    
    # Process dataset
    stats = extractor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preserve_structure=not args.no_structure,
        show_progress=not args.no_progress
    )
    
    # Final message
    if stats['total_patches'] > 0:
        print(f"\n🎉 Successfully extracted {stats['total_patches']} patches!")
        print(f"\n💡 Next steps:")
        print(f"   1. Train with patches: python orignal/train_model.py --data-dir {args.output_dir}")
        print(f"   2. Compare with original: Train both patched and non-patched versions")
        print(f"   3. Use smaller batch size due to increased dataset size")
    else:
        print(f"\n❌ No patches were extracted. Check input directory and quality settings.")

if __name__ == "__main__":
    main()
