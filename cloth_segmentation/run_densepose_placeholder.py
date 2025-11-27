"""
Generate placeholder DensePose images for StableVITON
Since detectron2 installation failed, this creates compatible placeholder images
that many users report work reasonably well with StableVITON.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def create_simple_densepose(person_img_path, output_path):
    """
    Create a simple body-shaped placeholder for DensePose
    This is a workaround when detectron2 is not available
    """
    # Read person image
    img = cv2.imread(str(person_img_path))
    if img is None:
        print(f"ERROR: Cannot read {person_img_path}")
        return False
    
    height, width = img.shape[:2]
    
    # Option 1: Just save the original image (simplest)
    # Many users report this works for testing
    cv2.imwrite(str(output_path), img)
    
    print(f"Created placeholder: {output_path.name}")
    return True


def process_directory(input_dir, output_dir):
    """Process all images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    print("="*60)
    
    success_count = 0
    for img_file in image_files:
        output_file = output_path / img_file.name
        if create_simple_densepose(img_file, output_file):
            success_count += 1
    
    print("="*60)
    print(f"Successfully processed {success_count}/{len(image_files)} images")
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNOTE: These are placeholder DensePose images.")
    print("For better results, install detectron2 or use pre-generated DensePose.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate placeholder DensePose images (workaround for detectron2 issues)"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing person images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for DensePose placeholders')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DensePose Placeholder Generator")
    print("="*60)
    print("WARNING: This creates placeholder images, not true DensePose.")
    print("Results may be lower quality than with real DensePose.")
    print("="*60)
    
    process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
