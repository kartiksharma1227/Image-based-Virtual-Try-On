"""
Simple Test Pipeline: Human Parsing → 4 VITON Outputs

This script:
1. Uses Ailia CLI to generate parsing maps
2. Converts parsing to 4 VITON formats
3. Visualizes results

Usage:
    python test_pipeline_simple.py
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path

# Add conversion script to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert_parsing_to_viton import (
    load_parsing_map,
    create_agnostic_parsing,
    create_agnostic_image,
    create_agnostic_mask,
    save_parsing_visualization
)

# Paths
AILIA_DIR = r"E:\RM Trial\dummy\ailia-models\image_segmentation\human_part_segmentation"
AILIA_SCRIPT = os.path.join(AILIA_DIR, "human_part_segmentation.py")


def run_ailia_cli(image_path, output_path, conda_env="viton-pip"):
    """
    Run Ailia human parsing using CLI
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🔍 Running Ailia parsing...")
    print(f"   Input: {os.path.basename(image_path)}")
    print(f"   Output: {os.path.basename(output_path)}")
    
    # Build command
    cmd = f'conda activate {conda_env} && cd "{AILIA_DIR}" && python human_part_segmentation.py -i "{image_path}" -s "{output_path}"'
    
    try:
        # Run command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"   ✅ Parsing complete!")
            return True
        else:
            print(f"   ❌ Parsing failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def convert_to_viton_format(person_path, parsing_path, output_root):
    """
    Convert Ailia parsing output to 4 VITON formats
    
    Returns:
        Dict with paths to generated files
    """
    print(f"\n🔄 Converting to VITON format...")
    
    base_name = os.path.splitext(os.path.basename(person_path))[0]
    
    # Create output directories
    output_dirs = {
        'parse_v3': os.path.join(output_root, 'image-parse-v3'),
        'parse_agnostic': os.path.join(output_root, 'image-parse-agnostic-v3.2'),
        'agnostic': os.path.join(output_root, 'agnostic-v3.2'),
        'agnostic_mask': os.path.join(output_root, 'agnostic-mask')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load images
    person_img = cv2.imread(person_path)
    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    
    # Load parsing map from Ailia output
    parsing = load_parsing_map(parsing_path)
    print(f"   Loaded parsing map: {parsing.shape}, unique labels: {len(np.unique(parsing))}")
    
    # 1. Save full parsing map (colorized)
    parse_v3_path = os.path.join(output_dirs['parse_v3'], f"{base_name}.png")
    save_parsing_visualization(parsing, parse_v3_path)
    print(f"   1️⃣  image-parse-v3: {os.path.basename(parse_v3_path)}")
    
    # 2. Create and save agnostic parsing
    agnostic_parsing = create_agnostic_parsing(parsing)
    parse_agnostic_path = os.path.join(output_dirs['parse_agnostic'], f"{base_name}.png")
    save_parsing_visualization(agnostic_parsing, parse_agnostic_path)
    print(f"   2️⃣  image-parse-agnostic-v3.2: {os.path.basename(parse_agnostic_path)}")
    
    # 3. Create and save agnostic image (ORIGINAL image with mask applied)
    agnostic_img = create_agnostic_image(person_rgb, parsing)
    agnostic_path = os.path.join(output_dirs['agnostic'], f"{base_name}.jpg")
    agnostic_bgr = cv2.cvtColor(agnostic_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(agnostic_path, agnostic_bgr)
    print(f"   3️⃣  agnostic-v3.2: {os.path.basename(agnostic_path)} ⭐ (ORIGINAL with upper clothes removed)")
    
    # 4. Create and save agnostic mask
    agnostic_mask = create_agnostic_mask(parsing)
    mask_path = os.path.join(output_dirs['agnostic_mask'], f"{base_name}_mask.png")
    cv2.imwrite(mask_path, agnostic_mask)
    print(f"   4️⃣  agnostic-mask: {os.path.basename(mask_path)}")
    
    return {
        'parse_v3': parse_v3_path,
        'parse_agnostic': parse_agnostic_path,
        'agnostic': agnostic_path,
        'agnostic_mask': mask_path
    }


def create_visualization_grid(person_path, parsing_path, outputs, save_path):
    """
    Create a visual comparison grid:
    Row 1: Original | Parse-v3 | Parse-Agnostic
    Row 2: Agnostic | Agnostic-Mask | Ailia-Output
    """
    print(f"\n📊 Creating visualization grid...")
    
    # Load images
    original = cv2.imread(person_path)
    parse_v3 = cv2.imread(outputs['parse_v3'])
    parse_agnostic = cv2.imread(outputs['parse_agnostic'])
    agnostic = cv2.imread(outputs['agnostic'])
    agnostic_mask = cv2.imread(outputs['agnostic_mask'])
    ailia_output = cv2.imread(parsing_path)
    
    # Convert mask to 3-channel for visualization
    if len(agnostic_mask.shape) == 2:
        agnostic_mask = cv2.cvtColor(agnostic_mask, cv2.COLOR_GRAY2BGR)
    
    # Resize all to same size
    target_size = (256, 384)  # Typical VITON size
    original = cv2.resize(original, target_size)
    parse_v3 = cv2.resize(parse_v3, target_size)
    parse_agnostic = cv2.resize(parse_agnostic, target_size)
    agnostic = cv2.resize(agnostic, target_size)
    agnostic_mask = cv2.resize(agnostic_mask, target_size)
    ailia_output = cv2.resize(ailia_output, target_size)
    
    # Add labels
    def add_label(img, text, color=(255, 255, 255)):
        img_copy = img.copy()
        # Black outline
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 3, cv2.LINE_AA)
        # White text
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2, cv2.LINE_AA)
        return img_copy
    
    original = add_label(original, "Original")
    parse_v3 = add_label(parse_v3, "Parse-v3")
    parse_agnostic = add_label(parse_agnostic, "Parse-Agnostic")
    agnostic = add_label(agnostic, "Agnostic-v3.2", (0, 255, 0))
    agnostic_mask = add_label(agnostic_mask, "Agnostic-Mask")
    ailia_output = add_label(ailia_output, "Ailia-Output")
    
    # Create grid
    row1 = np.hstack([original, parse_v3, parse_agnostic])
    row2 = np.hstack([agnostic, agnostic_mask, ailia_output])
    grid = np.vstack([row1, row2])
    
    # Save
    cv2.imwrite(save_path, grid)
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    
    return grid


def test_single_image(image_path, output_root):
    """
    Test pipeline on a single image
    """
    print(f"\n{'='*70}")
    print(f"📷 Processing: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create temp directory for Ailia output
    parsing_dir = os.path.join(output_root, '_parsing_temp')
    os.makedirs(parsing_dir, exist_ok=True)
    parsing_path = os.path.join(parsing_dir, f"{base_name}_parsing.png")
    
    # Step 1: Run Ailia parsing
    success = run_ailia_cli(image_path, parsing_path)
    if not success:
        print(f"❌ Failed to generate parsing map")
        return None
    
    # Step 2: Convert to VITON format
    outputs = convert_to_viton_format(image_path, parsing_path, output_root)
    
    # Step 3: Create visualization
    vis_dir = os.path.join(output_root, '_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, f"{base_name}_comparison.jpg")
    create_visualization_grid(image_path, parsing_path, outputs, vis_path)
    
    print(f"\n✅ Successfully processed: {os.path.basename(image_path)}")
    
    return {
        'image': image_path,
        'parsing': parsing_path,
        'outputs': outputs,
        'visualization': vis_path
    }


def main():
    print("="*70)
    print("🧪 TESTING PIPELINE: AILIA PARSING → 4 VITON FORMATS")
    print("="*70)
    
    # Get 3 test images
    test_dir = r"E:\RM Trial\StableVITON-master\StableVITON-master\test\image"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Get first 3 images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(Path(test_dir).glob(ext)))
        if len(test_images) >= 3:
            break
    
    test_images = [str(p) for p in test_images[:3]]
    
    if len(test_images) == 0:
        print(f"❌ No images found in: {test_dir}")
        return
    
    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")
    
    # Output directory
    output_root = r"E:\RM Trial\preprocessing_toolkit\test_pipeline_output"
    os.makedirs(output_root, exist_ok=True)
    
    # Process each image
    results = []
    for image_path in test_images:
        try:
            result = test_single_image(image_path, output_root)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 PIPELINE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {len(results)}/{len(test_images)}")
    print(f"❌ Failed: {len(test_images) - len(results)}/{len(test_images)}")
    
    if len(results) > 0:
        print(f"\n📁 Output Structure:")
        print(f"  {output_root}/")
        print(f"  ├── image-parse-v3/            # Full parsing maps")
        print(f"  ├── image-parse-agnostic-v3.2/ # Agnostic parsing")
        print(f"  ├── agnostic-v3.2/             # ⭐ ORIGINAL image with upper clothes removed")
        print(f"  ├── agnostic-mask/             # Binary masks")
        print(f"  ├── _parsing_temp/             # Ailia raw outputs")
        print(f"  └── _visualizations/           # Comparison grids")
        
        print(f"\n🎯 Key Output:")
        print(f"   agnostic-v3.2/ = ORIGINAL person image with mask applied (gray where clothes removed)")
    
    print(f"\n✅ PIPELINE TEST COMPLETE!")
    print(f"Check outputs in: {output_root}")


if __name__ == '__main__':
    main()
