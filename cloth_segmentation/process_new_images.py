"""
Complete Pipeline: Process new person + cloth images for StableVITON
This script takes raw person images and cloth images and generates all required inputs.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout[-500:])  # Last 500 chars
    else:
        print(f"❌ {description} - FAILED")
        print(result.stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process new images for StableVITON')
    parser.add_argument('--person_dir', type=str, required=True, help='Directory with person images')
    parser.add_argument('--cloth_dir', type=str, required=True, help='Directory with cloth images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory (StableVITON format)')
    parser.add_argument('--skip_cloth_seg', action='store_true', help='Skip cloth segmentation if masks exist')
    parser.add_argument('--skip_parsing', action='store_true', help='Skip human parsing if exists')
    
    args = parser.parse_args()
    
    # Paths
    preprocessing_root = Path(__file__).parent.parent.absolute()
    ailia_path = r"E:\RM Trial\dummy\ailia-models\image_segmentation\human_part_segmentation"
    
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "_temp"
    
    # Create output structure
    dirs_to_create = [
        temp_dir / "parsing_raw",
        output_dir / "image",
        output_dir / "cloth",
        output_dir / "cloth-mask",
        output_dir / "image-parse-v3",
        output_dir / "image-parse-agnostic-v3.2",
        output_dir / "agnostic-v3.2",
        output_dir / "agnostic-mask",
        output_dir / "image-densepose"  # placeholder
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting Complete Processing Pipeline")
    print(f"Person images: {args.person_dir}")
    print(f"Cloth images: {args.cloth_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Copy person images
    if not run_command(
        f'xcopy "{args.person_dir}" "{output_dir / "image"}" /E /I /Y',
        "Copy person images"
    ):
        return
    
    # Step 2: Copy cloth images
    if not run_command(
        f'xcopy "{args.cloth_dir}" "{output_dir / "cloth"}" /E /I /Y',
        "Copy cloth images"
    ):
        return
    
    # Step 3: Generate cloth masks using U-2-Net
    if not args.skip_cloth_seg:
        if not run_command(
            f'conda activate viton-pip && python "{preprocessing_root / "scripts" / "run_cloth_segmentation.py"}" '
            f'--input_dir "{output_dir / "cloth"}" '
            f'--output_dir "{output_dir / "cloth-mask"}" '
            f'--device cuda',
            "Generate cloth masks (U-2-Net)"
        ):
            return
    
    # Step 4: Generate human parsing using Ailia
    if not args.skip_parsing:
        person_images = (
            list((output_dir / "image").glob("*.jpg")) + 
            list((output_dir / "image").glob("*.png")) +
            list((output_dir / "image").glob("*.jpeg"))
        )
        
        print(f"\n{'='*60}")
        print(f"Step: Generate human parsing (Ailia)")
        print(f"{'='*60}")
        print(f"Processing {len(person_images)} person images...")
        
        for idx, img_path in enumerate(person_images, 1):
            output_path = temp_dir / "parsing_raw" / f"{img_path.stem}_parsing.png"
            
            cmd = (
                f'cd "{ailia_path}" && '
                f'conda activate viton-pip && '
                f'python human_part_segmentation.py -i "{img_path}" -s "{output_path}"'
            )
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  [{idx}/{len(person_images)}] ✅ {img_path.name}")
            else:
                print(f"  [{idx}/{len(person_images)}] ❌ {img_path.name}")
                print(f"  Error: {result.stderr}")
        
        print(f"✅ Human parsing complete")
    
    # Step 5: Convert parsing to VITON format (4 outputs)
    if not run_command(
        f'conda activate viton-pip && python "{preprocessing_root / "scripts" / "convert_parsing_to_viton.py"}" '
        f'--person_dir "{output_dir / "image"}" '
        f'--parsing_dir "{temp_dir / "parsing_raw"}" '
        f'--output_dir "{output_dir}"',
        "Convert parsing to VITON format"
    ):
        return
    
    # Step 6: Create placeholder DensePose images
    if not run_command(
        f'conda activate viton-pip && python "{preprocessing_root / "scripts" / "run_densepose_placeholder.py"}" '
        f'--input_dir "{output_dir / "image"}" '
        f'--output_dir "{output_dir / "image-densepose"}"',
        "Create placeholder DensePose images"
    ):
        return
    
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput structure created at: {output_dir}")
    print(f"\nGenerated folders:")
    print(f"  ✅ image/                      - Person images")
    print(f"  ✅ cloth/                      - Cloth images")
    print(f"  ✅ cloth-mask/                 - Cloth segmentation masks")
    print(f"  ✅ image-parse-v3/             - Full parsing maps")
    print(f"  ✅ image-parse-agnostic-v3.2/  - Agnostic parsing")
    print(f"  ✅ agnostic-v3.2/              - Person without clothes")
    print(f"  ✅ agnostic-mask/              - Body area masks")
    print(f"  ✅ image-densepose/            - DensePose (placeholder)")
    print(f"\nYou can now run StableVITON inference with:")
    print(f'  python inference.py --data_root_dir "{output_dir}" --save_dir "./outputs/my_results"')

if __name__ == "__main__":
    main()
