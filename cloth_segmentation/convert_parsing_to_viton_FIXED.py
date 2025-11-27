#!/usr/bin/env python3
"""
FIXED VERSION: Creates VITON-format agnostic masks that include ARMS
This allows full-sleeve clothes to be properly rendered.

Key Change: UPPER_BODY_LABELS now includes arms (14, 15) and optionally hands (3)
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

# LIP Parsing Labels (20-class)
LIP_LABELS = {
    0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Glove', 4: 'Sunglasses',
    5: 'UpperClothes', 6: 'Dress', 7: 'Coat', 8: 'Socks', 9: 'Pants',
    10: 'Jumpsuits', 11: 'Scarf', 12: 'Skirt', 13: 'Face', 14: 'Left-arm',
    15: 'Right-arm', 16: 'Left-leg', 17: 'Right-leg', 18: 'Left-shoe', 19: 'Right-shoe'
}

# ===== KEY FIX: Include arms in the agnostic area =====
# Original: only cloth labels
# UPPER_BODY_LABELS = [5, 6, 7]  

# FIXED: Include arms and optionally hands/gloves
UPPER_BODY_LABELS = [5, 6, 7, 14, 15]  # upper_clothes, dress, coat, left-arm, right-arm
# Optional: Add 3 (gloves) if you want hands to be changeable too
# UPPER_BODY_LABELS = [3, 5, 6, 7, 14, 15]

# Labels to keep visible
KEEP_LABELS = [0, 1, 2, 4, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]  # Everything except upper body + arms

def create_agnostic_mask(parsing, height, width):
    """
    Create agnostic-mask: BLACK for cloth+arms area, WHITE for preserved areas
    This allows the AI to generate full sleeves when needed
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Mark upper body + arms as WHITE (these will be BLACK in final mask)
    for label in UPPER_BODY_LABELS:
        mask[parsing == label] = 255
    
    # The mask output should be:
    # WHITE (255) = cloth + arms area (will be changed by AI)
    # BLACK (0) = face, legs, background (preserved)
    
    # But VITON expects inverse, so we return as-is and will save correctly
    return mask


def create_agnostic_v32(parsing, original_img, height, width):
    """
    Create agnostic-v3.2: Person with upper body + arms removed (filled with gray/noise)
    """
    agnostic = original_img.copy()
    
    # Create mask for areas to remove
    remove_mask = np.zeros((height, width), dtype=bool)
    for label in UPPER_BODY_LABELS:
        remove_mask[parsing == label] = True
    
    # Fill removed areas with gray (you can also use noise or inpainting)
    agnostic[remove_mask] = [128, 128, 128]  # Gray fill
    
    return agnostic


def create_parse_agnostic_v32(parsing, height, width):
    """
    Create image-parse-agnostic-v3.2: Parsing with upper body removed
    """
    parse_agnostic = parsing.copy()
    
    # Set upper body + arms to background (0)
    for label in UPPER_BODY_LABELS:
        parse_agnostic[parse_agnostic == label] = 0
    
    return parse_agnostic


def visualize_parsing(parsing, height, width):
    """
    Create colored visualization of parsing (image-parse-v3)
    """
    # Color palette for LIP labels (20 colors)
    palette = [
        0, 0, 0,       # 0: Background
        128, 0, 0,     # 1: Hat
        255, 0, 0,     # 2: Hair
        0, 85, 0,      # 3: Glove
        170, 0, 51,    # 4: Sunglasses
        255, 85, 0,    # 5: UpperClothes
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
    
    # Create RGB image
    parse_vis = Image.fromarray(parsing.astype(np.uint8))
    parse_vis.putpalette(palette)
    parse_vis = parse_vis.convert('RGB')
    
    return np.array(parse_vis)


def process_single_person(person_name, person_dir, parsing_dir, output_dir):
    """
    Process a single person: generate all VITON format outputs
    """
    # Paths
    person_path = Path(person_dir) / f"{person_name}.jpg"
    parsing_path = Path(parsing_dir) / f"{person_name}_parsing.png"
    
    if not person_path.exists():
        print(f"[SKIP] Person image not found: {person_path}")
        return False
    
    if not parsing_path.exists():
        print(f"[SKIP] Parsing not found: {parsing_path}")
        return False
    
    # Read inputs
    person_img = cv2.imread(str(person_path))
    person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    parsing = np.array(Image.open(parsing_path))
    
    # Always resize both to target dimensions
    person_img = cv2.resize(person_img, (768, 1024), interpolation=cv2.INTER_LANCZOS4)
    parsing = cv2.resize(parsing, (768, 1024), interpolation=cv2.INTER_NEAREST)
    height, width = 1024, 768
    
    # Create output directories
    output_path = Path(output_dir)
    (output_path / "agnostic-v3.2").mkdir(parents=True, exist_ok=True)
    (output_path / "agnostic-mask").mkdir(parents=True, exist_ok=True)
    (output_path / "image-parse-v3").mkdir(parents=True, exist_ok=True)
    (output_path / "image-parse-agnostic-v3.2").mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    agnostic_mask = create_agnostic_mask(parsing, height, width)
    agnostic_v32 = create_agnostic_v32(parsing, person_img, height, width)
    parse_agnostic_v32 = create_parse_agnostic_v32(parsing, height, width)
    parse_v3 = visualize_parsing(parsing, height, width)
    
    # Save outputs
    Image.fromarray(agnostic_mask).save(output_path / "agnostic-mask" / f"{person_name}_mask.png")
    Image.fromarray(agnostic_v32).save(output_path / "agnostic-v3.2" / f"{person_name}.jpg")
    Image.fromarray(parse_agnostic_v32).save(output_path / "image-parse-agnostic-v3.2" / f"{person_name}.png")
    Image.fromarray(parse_v3).save(output_path / "image-parse-v3" / f"{person_name}.png")
    
    # Debug: Count cloth pixels
    cloth_pixels = np.sum(agnostic_mask == 255)
    print(f"[OK] {person_name}: {cloth_pixels} cloth+arm pixels marked as changeable")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert Ailia parsing to VITON format (FIXED with arms)")
    parser.add_argument("--person_dir", type=str, required=True, help="Directory with person images")
    parser.add_argument("--parsing_dir", type=str, required=True, help="Directory with Ailia parsing outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for VITON format")
    args = parser.parse_args()
    
    print(f"\n[INFO] Processing with FIXED agnostic mask (includes arms!)")
    print(f"[INFO] Upper body labels: {UPPER_BODY_LABELS}")
    print(f"[INFO] This allows full-sleeve clothes to be rendered properly\n")
    
    # Get all person images
    person_files = sorted(Path(args.person_dir).glob("*.jpg"))
    
    success_count = 0
    for person_file in person_files:
        person_name = person_file.stem
        if process_single_person(person_name, args.person_dir, args.parsing_dir, args.output_dir):
            success_count += 1
    
    print(f"\n[SUCCESS] Successfully processed {success_count}/{len(person_files)} images")
    print(f"[SUCCESS] CONVERSION COMPLETE WITH ARM INCLUSION!")
    
    return 0


if __name__ == "__main__":
    exit(main())
