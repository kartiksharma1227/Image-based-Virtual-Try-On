"""
Test Full Pipeline: Human Parsing → 4 VITON Outputs

This script tests the complete workflow:
1. Run Ailia human parsing on test images
2. Convert parsing to 4 VITON formats
3. Visualize and verify outputs
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import shutil
from pathlib import Path

# Add paths
AILIA_PATH = r"E:\RM Trial\dummy\ailia-models\image_segmentation\human_part_segmentation"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AILIA_PATH)
sys.path.insert(0, SCRIPT_DIR)

# Import ailia utilities
try:
    import ailia
except ImportError:
    print("Warning: ailia not found. Make sure you have ailia SDK installed.")
    sys.exit(1)

# Import our conversion script
from convert_parsing_to_viton import (
    load_parsing_map,
    create_agnostic_parsing,
    create_agnostic_image,
    create_agnostic_mask,
    save_parsing_visualization,
    get_palette
)


def run_ailia_parsing(image_path, output_dir):
    """
    Run Ailia human parsing on a single image
    
    Returns:
        parsing: numpy array (H, W) with label values 0-19
        parsing_path: path to saved parsing visualization
    """
    print(f"\n🔍 Running Ailia parsing on: {os.path.basename(image_path)}")
    
    # Load model
    model_path = os.path.join(AILIA_PATH, "resnet-lip.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"  Loading model: {model_path}")
    net = ailia.Net(model_path, env_id=ailia.get_gpu_environment_id())
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size (473x473 for LIP)
    input_size = (473, 473)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    mean = np.array([123.68, 116.779, 103.939])
    img_normalized = img_resized - mean
    
    # Add batch dimension and transpose to NCHW
    img_input = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    img_input = np.expand_dims(img_input, 0).astype(np.float32)  # Add batch
    
    # Run inference
    print(f"  Running inference...")
    output = net.predict({'data': img_input})
    
    # Get parsing result
    if isinstance(output, dict):
        # Find the output tensor (usually 'fc1_interp' or similar)
        if 'fc1_interp' in output:
            parsing_logits = output['fc1_interp']
        else:
            # Take first output
            parsing_logits = list(output.values())[0]
    else:
        parsing_logits = output
    
    # Get label map from logits (argmax over classes)
    if len(parsing_logits.shape) == 4:  # (N, C, H, W)
        parsing_labels = np.argmax(parsing_logits[0], axis=0)
    else:
        parsing_labels = np.argmax(parsing_logits, axis=0)
    
    # Resize back to original size
    parsing = cv2.resize(
        parsing_labels.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Save parsing visualization
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    parsing_path = os.path.join(output_dir, f"{base_name}_parsing.png")
    
    save_parsing_visualization(parsing, parsing_path)
    print(f"  ✅ Saved parsing: {parsing_path}")
    
    return parsing, parsing_path


def convert_to_viton_format(person_path, parsing, output_root):
    """
    Convert parsing to 4 VITON formats
    
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
    
    # Load person image
    person_img = cv2.imread(person_path)
    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    
    # 1. Save full parsing map (colorized)
    parse_v3_path = os.path.join(output_dirs['parse_v3'], f"{base_name}.png")
    save_parsing_visualization(parsing, parse_v3_path)
    print(f"  1️⃣  image-parse-v3: {os.path.basename(parse_v3_path)}")
    
    # 2. Create and save agnostic parsing
    agnostic_parsing = create_agnostic_parsing(parsing)
    parse_agnostic_path = os.path.join(output_dirs['parse_agnostic'], f"{base_name}.png")
    save_parsing_visualization(agnostic_parsing, parse_agnostic_path)
    print(f"  2️⃣  image-parse-agnostic-v3.2: {os.path.basename(parse_agnostic_path)}")
    
    # 3. Create and save agnostic image (ORIGINAL image with mask applied)
    agnostic_img = create_agnostic_image(person_rgb, parsing)
    agnostic_path = os.path.join(output_dirs['agnostic'], f"{base_name}.jpg")
    agnostic_bgr = cv2.cvtColor(agnostic_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(agnostic_path, agnostic_bgr)
    print(f"  3️⃣  agnostic-v3.2: {os.path.basename(agnostic_path)} (original image with upper clothes removed)")
    
    # 4. Create and save agnostic mask
    agnostic_mask = create_agnostic_mask(parsing)
    mask_path = os.path.join(output_dirs['agnostic_mask'], f"{base_name}_mask.png")
    cv2.imwrite(mask_path, agnostic_mask)
    print(f"  4️⃣  agnostic-mask: {os.path.basename(mask_path)}")
    
    return {
        'parse_v3': parse_v3_path,
        'parse_agnostic': parse_agnostic_path,
        'agnostic': agnostic_path,
        'agnostic_mask': mask_path
    }


def create_visualization_grid(person_path, outputs, save_path):
    """
    Create a visual comparison grid:
    Row 1: Original | Parse-v3 | Parse-Agnostic
    Row 2: Agnostic | Agnostic-Mask | Combined
    """
    print(f"\n📊 Creating visualization grid...")
    
    # Load images
    original = cv2.imread(person_path)
    parse_v3 = cv2.imread(outputs['parse_v3'])
    parse_agnostic = cv2.imread(outputs['parse_agnostic'])
    agnostic = cv2.imread(outputs['agnostic'])
    agnostic_mask = cv2.imread(outputs['agnostic_mask'])
    
    # Convert mask to 3-channel for visualization
    if len(agnostic_mask.shape) == 2:
        agnostic_mask = cv2.cvtColor(agnostic_mask, cv2.COLOR_GRAY2BGR)
    
    # Create combined view (original with agnostic mask overlay)
    combined = original.copy()
    mask_overlay = cv2.addWeighted(combined, 0.6, agnostic_mask, 0.4, 0)
    
    # Resize all to same size
    target_size = (256, 384)  # Typical VITON size
    original = cv2.resize(original, target_size)
    parse_v3 = cv2.resize(parse_v3, target_size)
    parse_agnostic = cv2.resize(parse_agnostic, target_size)
    agnostic = cv2.resize(agnostic, target_size)
    agnostic_mask = cv2.resize(agnostic_mask, target_size)
    mask_overlay = cv2.resize(mask_overlay, target_size)
    
    # Add labels
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1, cv2.LINE_AA)
        return img_copy
    
    original = add_label(original, "Original")
    parse_v3 = add_label(parse_v3, "Parse-v3")
    parse_agnostic = add_label(parse_agnostic, "Parse-Agnostic")
    agnostic = add_label(agnostic, "Agnostic-v3.2")
    agnostic_mask = add_label(agnostic_mask, "Agnostic-Mask")
    mask_overlay = add_label(mask_overlay, "Mask Overlay")
    
    # Create grid
    row1 = np.hstack([original, parse_v3, parse_agnostic])
    row2 = np.hstack([agnostic, agnostic_mask, mask_overlay])
    grid = np.vstack([row1, row2])
    
    # Save
    cv2.imwrite(save_path, grid)
    print(f"  ✅ Saved visualization: {save_path}")
    
    return grid


def test_pipeline(test_images, output_root):
    """
    Run full pipeline on test images
    """
    print("="*60)
    print("🧪 TESTING FULL PIPELINE: HUMAN PARSING → VITON FORMATS")
    print("="*60)
    
    os.makedirs(output_root, exist_ok=True)
    
    results = []
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"\n{'='*60}")
        print(f"📷 Test Image {idx}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Run Ailia parsing
            parsing_dir = os.path.join(output_root, 'parsing_outputs')
            parsing, parsing_path = run_ailia_parsing(image_path, parsing_dir)
            
            # Step 2: Convert to VITON format
            outputs = convert_to_viton_format(image_path, parsing, output_root)
            
            # Step 3: Create visualization
            vis_dir = os.path.join(output_root, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = os.path.join(vis_dir, f"{base_name}_comparison.jpg")
            create_visualization_grid(image_path, outputs, vis_path)
            
            results.append({
                'image': image_path,
                'parsing': parsing_path,
                'outputs': outputs,
                'visualization': vis_path,
                'success': True
            })
            
            print(f"\n✅ Successfully processed: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image': image_path,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PIPELINE TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Successful: {successful}/{len(test_images)}")
    print(f"❌ Failed: {len(test_images) - successful}/{len(test_images)}")
    
    if successful > 0:
        print(f"\n📁 Output Structure:")
        print(f"  {output_root}/")
        print(f"  ├── parsing_outputs/        # Ailia parsing results")
        print(f"  ├── image-parse-v3/         # Full parsing maps")
        print(f"  ├── image-parse-agnostic-v3.2/  # Agnostic parsing")
        print(f"  ├── agnostic-v3.2/          # Agnostic images (ORIGINAL with mask)")
        print(f"  ├── agnostic-mask/          # Binary masks")
        print(f"  └── visualizations/         # Comparison grids")
    
    return results


def main():
    # Get test images from StableVITON test folder
    stableviton_test = r"E:\RM Trial\StableVITON-master\StableVITON-master\test\image"
    
    if not os.path.exists(stableviton_test):
        print(f"❌ Test directory not found: {stableviton_test}")
        print("Please update the path or provide test images.")
        return
    
    # Get first 3 images for testing
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(Path(stableviton_test).glob(ext))[:3])
    
    if len(test_images) == 0:
        print(f"❌ No images found in: {stableviton_test}")
        return
    
    test_images = [str(p) for p in test_images[:3]]  # Limit to 3 images
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")
    
    # Run pipeline
    output_root = r"E:\RM Trial\preprocessing_toolkit\test_pipeline_output"
    results = test_pipeline(test_images, output_root)
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"Check outputs in: {output_root}")


if __name__ == '__main__':
    main()
