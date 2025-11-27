"""
Resize all images in a dataset to StableVITON's expected size (768x1024)
"""
import os
import sys
from pathlib import Path
from PIL import Image
import argparse

def resize_image(input_path, output_path, target_size=(768, 1024)):
    """
    Resize image to target size
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_size: (width, height) tuple
    """
    img = Image.open(input_path)
    
    # Resize to target size
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    # Save with same format
    img_resized.save(output_path, quality=95)
    print(f"  Resized: {Path(input_path).name} from {img.size} to {img_resized.size}")

def resize_directory(directory, target_size=(768, 1024)):
    """Resize all images in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))
    
    if not files:
        print(f"No images found in {directory}")
        return
    
    print(f"\nResizing {len(files)} images in {directory.name}/")
    
    for img_path in files:
        resize_image(img_path, img_path, target_size)

def main():
    parser = argparse.ArgumentParser(description='Resize dataset images to StableVITON size')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing test/train folders')
    parser.add_argument('--width', type=int, default=768,
                        help='Target width (default: 768)')
    parser.add_argument('--height', type=int, default=1024,
                        help='Target height (default: 1024)')
    
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    data_root = Path(args.data_root)
    
    print(f"="*60)
    print(f"Resizing all images to {target_size[0]}x{target_size[1]}")
    print(f"="*60)
    
    # Process test directory
    test_dir = data_root / "test"
    if test_dir.exists():
        subdirs = [
            "image",
            "cloth",
            "agnostic-v3.2",
            "agnostic-mask",
            "image-parse-v3",
            "image-parse-agnostic-v3.2",
            "cloth-mask",
            "image-densepose"
        ]
        
        for subdir in subdirs:
            dir_path = test_dir / subdir
            if dir_path.exists():
                resize_directory(dir_path, target_size)
    
    # Process train directory if exists
    train_dir = data_root / "train"
    if train_dir.exists():
        subdirs = [
            "image",
            "cloth",
            "agnostic-v3.2",
            "agnostic-mask",
            "image-parse-v3",
            "image-parse-agnostic-v3.2",
            "cloth-mask",
            "image-densepose"
        ]
        
        for subdir in subdirs:
            dir_path = train_dir / subdir
            if dir_path.exists():
                resize_directory(dir_path, target_size)
    
    print(f"\n{'='*60}")
    print(f"Resizing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
