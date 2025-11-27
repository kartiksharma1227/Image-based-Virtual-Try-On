"""
Convert Human Parsing outputs to StableVITON format
Generates 4 required outputs from parsing map:
1. image-parse-v3/ - Full parsing map (colorized visualization)
2. image-parse-agnostic-v3.2/ - Agnostic parsing (upper clothes removed)
3. agnostic-v3.2/ - Person image without upper clothes
4. agnostic-mask/ - Binary mask of body area to preserve

LIP (Look Into Person) Dataset Labels:
0: Background
1: Hat
2: Hair
3: Glove
4: Sunglasses
5: Upper-clothes
6: Dress
7: Coat
8: Socks
9: Pants
10: Jumpsuits
11: Scarf
12: Skirt
13: Face
14: Left-arm
15: Right-arm
16: Left-leg
17: Right-leg
18: Left-shoe
19: Right-shoe
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm

# Add Ailia path to import their utilities if needed
AILIA_PATH = r"E:\RM Trial\dummy\ailia-models\image_segmentation\human_part_segmentation"
if AILIA_PATH not in sys.path:
    sys.path.insert(0, AILIA_PATH)


# LIP label definitions
LABELS = {
    'background': 0,
    'hat': 1,
    'hair': 2,
    'glove': 3,
    'sunglasses': 4,
    'upper_clothes': 5,
    'dress': 6,
    'coat': 7,
    'socks': 8,
    'pants': 9,
    'jumpsuits': 10,
    'scarf': 11,
    'skirt': 12,
    'face': 13,
    'left_arm': 14,
    'right_arm': 15,
    'left_leg': 16,
    'right_leg': 17,
    'left_shoe': 18,
    'right_shoe': 19
}

# Parts to REMOVE for agnostic image (upper body clothing)
UPPER_BODY_LABELS = [
    LABELS['upper_clothes'],  # 5
    LABELS['coat'],          # 7
    LABELS['dress'],         # 6
]

# Parts to SHOW AS WHITE in agnostic mask
# The mask should be WHITE (255) for the ENTIRE BODY (including torso/arms)
# and BLACK (0) ONLY for the cloth itself (upper_clothes/coat/dress)
# This shows the area where the cloth will overlay
PRESERVE_LABELS = [
    LABELS['background'],    # 0 - keep background white
    LABELS['hat'],          # 1
    LABELS['hair'],         # 2
    LABELS['sunglasses'],   # 4
    LABELS['face'],         # 13
    LABELS['pants'],        # 9
    LABELS['skirt'],        # 12
    LABELS['left_arm'],     # 14
    LABELS['right_arm'],    # 15
    LABELS['left_leg'],     # 16
    LABELS['right_leg'],    # 17
    LABELS['left_shoe'],    # 18
    LABELS['right_shoe'],   # 19
    # The torso/arms stay WHITE - only the CLOTH label areas become BLACK
]


def get_palette():
    """
    Get color palette for visualization (LIP dataset standard)
    Returns palette array for 20 labels
    """
    palette = [
        0, 0, 0,       # 0: Background
        128, 0, 0,     # 1: Hat
        255, 0, 0,     # 2: Hair
        0, 85, 0,      # 3: Glove
        170, 0, 51,    # 4: Sunglasses
        255, 85, 0,    # 5: Upper-clothes
        0, 0, 85,      # 6: Dress
        0, 119, 221,   # 7: Coat
        85, 85, 0,     # 8: Socks
        0, 85, 85,     # 9: Pants
        85, 51, 0,     # 10: Jumpsuits
        52, 86, 128,   # 11: Scarf
        0, 128, 0,     # 12: Skirt
        0, 0, 255,     # 13: Face
        51, 170, 221,  # 14: Left-arm
        0, 255, 255,   # 15: Right-arm
        85, 255, 170,  # 16: Left-leg
        170, 255, 85,  # 17: Right-leg
        255, 255, 0,   # 18: Left-shoe
        255, 170, 0    # 19: Right-shoe
    ]
    return palette


def run_ailia_parsing(image_path, output_path):
    """
    Run Ailia human parsing on an image
    Returns: path to generated parsing map
    """
    try:
        # Import ailia model
        import ailia
        from human_part_segmentation import predict, preprocess, postprocess
        
        # Load model
        model_path = os.path.join(AILIA_PATH, "resnet-lip.onnx")
        net = ailia.Net(model_path, env_id=ailia.get_gpu_environment_id())
        
        # Run inference
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get parsing result
        parsing = predict(net, img_rgb)
        
        # Save parsing map (as grayscale label map)
        parsing_gray = Image.fromarray(parsing.astype(np.uint8), mode='P')
        parsing_gray.putpalette(get_palette())
        parsing_gray.save(output_path)
        
        return parsing
        
    except Exception as e:
        print(f"Error running Ailia parsing: {e}")
        print("Falling back to loading existing parsing map...")
        return None


def load_parsing_map(parsing_path):
    """
    Load parsing map (can be grayscale label map or colored visualization)
    Returns: numpy array of shape (H, W) with label values 0-19
    """
    img = Image.open(parsing_path)
    
    # If RGB/RGBA, convert to labels (assuming it's a colored visualization)
    if img.mode in ['RGB', 'RGBA']:
        img_array = np.array(img.convert('RGB'))
        # Create reverse mapping from colors to labels
        palette = get_palette()
        parsing = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        
        for label_id in range(20):
            r, g, b = palette[label_id*3:label_id*3+3]
            mask = (img_array[:,:,0] == r) & (img_array[:,:,1] == g) & (img_array[:,:,2] == b)
            parsing[mask] = label_id
            
    # If already grayscale palette image
    elif img.mode == 'P':
        parsing = np.array(img)
    
    # If grayscale
    elif img.mode == 'L':
        parsing = np.array(img)
    
    else:
        raise ValueError(f"Unsupported image mode: {img.mode}")
    
    return parsing


def create_agnostic_parsing(parsing):
    """
    Create agnostic parsing map (remove upper clothes labels)
    Replace upper clothes with background or arms
    """
    agnostic_parsing = parsing.copy()
    
    # Replace upper body clothing with background (0)
    for label in UPPER_BODY_LABELS:
        agnostic_parsing[parsing == label] = LABELS['background']
    
    return agnostic_parsing


def create_agnostic_image(person_image, parsing):
    """
    Create agnostic image (ORIGINAL person image with upper clothes removed)
    This is the ORIGINAL RGB image with upper body clothing masked out
    Fill removed areas with gray (128, 128, 128) or use inpainting
    
    NOTE: StableVITON expects this to be the ORIGINAL person image with masking,
    NOT the parsing map visualization!
    NOTE: parsing should already be resized to match person_image dimensions
    """
    # Start with ORIGINAL person image (RGB format)
    agnostic_img = person_image.copy()
    img_h, img_w = person_image.shape[:2]
    
    # Create mask of upper body parts to remove
    upper_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for label in UPPER_BODY_LABELS:
        upper_mask[parsing == label] = 255
    
    # Apply mask to ORIGINAL image: fill upper clothes area with gray
    agnostic_img[upper_mask == 255] = [128, 128, 128]
    
    # Alternative: Use inpainting for smoother results
    # agnostic_img_bgr = cv2.cvtColor(agnostic_img, cv2.COLOR_RGB2BGR)
    # agnostic_img_bgr = cv2.inpaint(agnostic_img_bgr, upper_mask, 3, cv2.INPAINT_TELEA)
    # agnostic_img = cv2.cvtColor(agnostic_img_bgr, cv2.COLOR_BGR2RGB)
    
    return agnostic_img


def create_agnostic_mask(parsing):
    """
    Create agnostic mask (areas where cloth will be placed)
    Binary mask: 
    - BLACK (0) = Background and non-cloth areas
    - WHITE (255) = CLOTH area (upper_clothes, coat, dress) - this is where new cloth goes
    
    The WHITE area shows the CLOTH REGION for warping
    """
    h, w = parsing.shape
    
    # Start with all BLACK (0)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark CLOTH areas as WHITE (255)
    cloth_pixels = 0
    for label in UPPER_BODY_LABELS:
        pixels_changed = np.sum(parsing == label)
        cloth_pixels += pixels_changed
        mask[parsing == label] = 255
    
    print(f"  [DEBUG] Agnostic mask: {cloth_pixels} cloth pixels marked as WHITE, rest is BLACK")
    
    return mask


def save_parsing_visualization(parsing, output_path):
    """
    Save parsing map as colored visualization
    """
    img = Image.fromarray(parsing.astype(np.uint8), mode='P')
    img.putpalette(get_palette())
    img.save(output_path)


def process_single_image(person_path, parsing_path, output_dirs, run_ailia=False):
    """
    Process a single person image and its parsing map
    
    Args:
        person_path: Path to original person image
        parsing_path: Path to parsing map (or None if run_ailia=True)
        output_dirs: Dict with keys: parse_v3, parse_agnostic, agnostic, agnostic_mask
        run_ailia: If True, run Ailia parsing first
    """
    person_name = os.path.basename(person_path)
    base_name = os.path.splitext(person_name)[0]
    
    # Load person image
    person_img = cv2.imread(person_path)
    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = person_rgb.shape[:2]
    
    # Get parsing map
    if run_ailia:
        # Run Ailia parsing
        temp_parsing_path = os.path.join(output_dirs['parse_v3'], f"{base_name}_temp.png")
        parsing = run_ailia_parsing(person_path, temp_parsing_path)
        if parsing is None:
            raise ValueError(f"Failed to run Ailia parsing on {person_path}")
    else:
        # Load existing parsing
        if parsing_path is None or not os.path.exists(parsing_path):
            raise ValueError(f"Parsing map not found: {parsing_path}")
        parsing = load_parsing_map(parsing_path)
    
    # Resize parsing to match person image dimensions if needed
    parse_h, parse_w = parsing.shape
    if (parse_h != img_h) or (parse_w != img_w):
        print(f"  [INFO] Resizing parsing from {parse_w}x{parse_h} to {img_w}x{img_h}")
        parsing = cv2.resize(parsing, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # 1. Save full parsing map (colorized)
    parse_v3_path = os.path.join(output_dirs['parse_v3'], f"{base_name}.png")
    save_parsing_visualization(parsing, parse_v3_path)
    
    # 2. Create and save agnostic parsing
    agnostic_parsing = create_agnostic_parsing(parsing)
    parse_agnostic_path = os.path.join(output_dirs['parse_agnostic'], f"{base_name}.png")
    save_parsing_visualization(agnostic_parsing, parse_agnostic_path)
    
    # 3. Create and save agnostic image
    agnostic_img = create_agnostic_image(person_rgb, parsing)
    agnostic_path = os.path.join(output_dirs['agnostic'], f"{base_name}.jpg")
    agnostic_bgr = cv2.cvtColor(agnostic_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(agnostic_path, agnostic_bgr)
    
    # 4. Create and save agnostic mask
    agnostic_mask = create_agnostic_mask(parsing)
    mask_path = os.path.join(output_dirs['agnostic_mask'], f"{base_name}_mask.png")
    cv2.imwrite(mask_path, agnostic_mask)
    
    return {
        'parse_v3': parse_v3_path,
        'parse_agnostic': parse_agnostic_path,
        'agnostic': agnostic_path,
        'agnostic_mask': mask_path
    }


def process_directory(person_dir, parsing_dir, output_root, run_ailia=False):
    """
    Process entire directory of person images
    
    Args:
        person_dir: Directory containing person images
        parsing_dir: Directory containing parsing maps (if run_ailia=False)
        output_root: Root directory for outputs
        run_ailia: If True, run Ailia parsing on each image
    """
    # Create output directories
    output_dirs = {
        'parse_v3': os.path.join(output_root, 'image-parse-v3'),
        'parse_agnostic': os.path.join(output_root, 'image-parse-agnostic-v3.2'),
        'agnostic': os.path.join(output_root, 'agnostic-v3.2'),
        'agnostic_mask': os.path.join(output_root, 'agnostic-mask')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all person images
    person_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        person_files.extend(Path(person_dir).glob(ext))
    
    print(f"Found {len(person_files)} person images to process")
    
    # Process each image
    results = []
    for person_path in tqdm(person_files, desc="Processing images"):
        try:
            # Find corresponding parsing map
            person_name = person_path.stem
            parsing_path = None
            
            if not run_ailia:
                # Look for parsing map with same name or with _parsing suffix
                for ext in ['.png', '.jpg', '.jpeg']:
                    # Try with _parsing suffix first (Ailia format)
                    test_path = os.path.join(parsing_dir, f"{person_name}_parsing{ext}")
                    if os.path.exists(test_path):
                        parsing_path = test_path
                        break
                    # Try without suffix
                    test_path = os.path.join(parsing_dir, f"{person_name}{ext}")
                    if os.path.exists(test_path):
                        parsing_path = test_path
                        break
            
            result = process_single_image(
                str(person_path),
                parsing_path,
                output_dirs,
                run_ailia=run_ailia
            )
            results.append((str(person_path), result))
            
        except Exception as e:
            print(f"\nError processing {person_path}: {e}")
            continue
    
    print(f"\n[SUCCESS] Successfully processed {len(results)}/{len(person_files)} images")
    print(f"Outputs saved to: {output_root}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Convert human parsing to StableVITON format')
    parser.add_argument('--person_dir', type=str, required=True,
                        help='Directory containing person images')
    parser.add_argument('--parsing_dir', type=str, default=None,
                        help='Directory containing parsing maps (if already generated)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root directory')
    parser.add_argument('--run_ailia', action='store_true',
                        help='Run Ailia parsing on each image (if parsing_dir not provided)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.person_dir):
        raise ValueError(f"Person directory not found: {args.person_dir}")
    
    if not args.run_ailia and not args.parsing_dir:
        raise ValueError("Must provide either --parsing_dir or --run_ailia")
    
    if args.parsing_dir and not os.path.exists(args.parsing_dir):
        raise ValueError(f"Parsing directory not found: {args.parsing_dir}")
    
    # Process images
    results = process_directory(
        args.person_dir,
        args.parsing_dir,
        args.output_dir,
        run_ailia=args.run_ailia
    )
    
    print("\n" + "="*50)
    print("[SUCCESS] CONVERSION COMPLETE!")
    print("="*50)
    print(f"Generated outputs in:")
    print(f"  - {os.path.join(args.output_dir, 'image-parse-v3')}")
    print(f"  - {os.path.join(args.output_dir, 'image-parse-agnostic-v3.2')}")
    print(f"  - {os.path.join(args.output_dir, 'agnostic-v3.2')}")
    print(f"  - {os.path.join(args.output_dir, 'agnostic-mask')}")


if __name__ == '__main__':
    main()
