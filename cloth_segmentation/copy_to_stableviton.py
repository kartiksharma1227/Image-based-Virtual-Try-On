"""
Copy preprocessed outputs to StableVITON test/train directory
"""

import os
import shutil
import argparse
from pathlib import Path


def copy_files(src_dir, dst_dir, file_pattern="*", verbose=True):
    """Copy files from source to destination"""
    os.makedirs(dst_dir, exist_ok=True)
    
    src_path = Path(src_dir)
    files = list(src_path.glob(file_pattern))
    
    if len(files) == 0:
        print(f"⚠ No files found in: {src_dir}")
        return 0
    
    copied_count = 0
    for file in files:
        dst_file = Path(dst_dir) / file.name
        shutil.copy2(file, dst_file)
        if verbose:
            print(f"  Copied: {file.name}")
        copied_count += 1
    
    return copied_count


def main():
    parser = argparse.ArgumentParser(
        description="Copy OpenPose outputs to StableVITON directory"
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../output',
        help='Source directory with preprocessing outputs (default: ../output)'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        help='Target StableVITON data directory (e.g., ../../StableVITON-master/StableVITON-master/test)'
    )
    parser.add_argument(
        '--copy_type',
        type=str,
        default='all',
        choices=['all', 'openpose', 'parsing', 'densepose', 'cloth'],
        help='What to copy (default: all)'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(__file__).parent / args.source_dir
    if not source_dir.is_absolute():
        source_dir = (Path(__file__).parent / args.source_dir).resolve()
    
    target_dir = Path(args.target_dir)
    if not target_dir.is_absolute():
        target_dir = (Path(__file__).parent / args.target_dir).resolve()
    
    print("="*60)
    print("Copy Preprocessing Outputs to StableVITON")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Type: {args.copy_type}")
    print("="*60)
    
    if not source_dir.exists():
        print(f"\n✗ Source directory not found: {source_dir}")
        return
    
    if not target_dir.exists():
        print(f"\n✗ Target directory not found: {target_dir}")
        print("Please provide the correct path to StableVITON test or train directory")
        return
    
    total_copied = 0
    
    # Copy OpenPose files
    if args.copy_type in ['all', 'openpose']:
        print("\n[OpenPose]")
        
        # JSON files
        src_json = source_dir / "openpose_json"
        dst_json = target_dir / "openpose_json"
        if src_json.exists():
            count = copy_files(src_json, dst_json)
            total_copied += count
            print(f"✓ Copied {count} JSON files to: {dst_json}")
        
        # Image files
        src_img = source_dir / "openpose_img"
        dst_img = target_dir / "openpose_img"
        if src_img.exists():
            count = copy_files(src_img, dst_img)
            total_copied += count
            print(f"✓ Copied {count} image files to: {dst_img}")
    
    # Copy Human Parsing files (if available)
    if args.copy_type in ['all', 'parsing']:
        print("\n[Human Parsing]")
        src_parsing = source_dir / "image-parse-v3"
        dst_parsing = target_dir / "image-parse-v3"
        if src_parsing.exists():
            count = copy_files(src_parsing, dst_parsing)
            total_copied += count
            print(f"✓ Copied {count} parsing files to: {dst_parsing}")
        else:
            print("⚠ Parsing output not found (not yet processed)")
    
    # Copy DensePose files (if available)
    if args.copy_type in ['all', 'densepose']:
        print("\n[DensePose]")
        src_densepose = source_dir / "image-densepose"
        dst_densepose = target_dir / "image-densepose"
        if src_densepose.exists():
            count = copy_files(src_densepose, dst_densepose)
            total_copied += count
            print(f"✓ Copied {count} densepose files to: {dst_densepose}")
        else:
            print("⚠ DensePose output not found (not yet processed)")
    
    # Copy Cloth Mask files (if available)
    if args.copy_type in ['all', 'cloth']:
        print("\n[Cloth Masks]")
        src_cloth = source_dir / "cloth-mask"
        dst_cloth = target_dir / "cloth-mask"
        if src_cloth.exists():
            count = copy_files(src_cloth, dst_cloth)
            total_copied += count
            print(f"✓ Copied {count} cloth mask files to: {dst_cloth}")
        else:
            print("⚠ Cloth mask output not found (not yet processed)")
    
    print("\n" + "="*60)
    print(f"✓ Total files copied: {total_copied}")
    print("="*60)


if __name__ == '__main__':
    main()
