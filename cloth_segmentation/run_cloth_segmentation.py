"""
Cloth Mask Generation using U-2-Net
Generates binary masks for garment images for StableVITON

This script uses U-2-Net for salient object detection to extract cloth masks
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from torchvision import transforms
from torch.autograd import Variable

# Add U-2-Net to path
SCRIPT_DIR = Path(__file__).parent
U2NET_DIR = SCRIPT_DIR.parent / "cloth_segmentation"
sys.path.insert(0, str(U2NET_DIR))

# Import U2NET model
from model import U2NET, U2NETP


def load_model(model_path, model_type='u2net', device='cuda'):
    """Load U-2-Net model"""
    print(f"[INFO] Loading {model_type} model from {model_path}")
    
    if model_type == 'u2netp':
        net = U2NETP(3, 1)
    else:
        net = U2NET(3, 1)
    
    if device == 'cuda' and torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    net.eval()
    print("[INFO] Model loaded successfully!")
    
    return net


def normalize_output(d):
    """Normalize model output to [0, 1]"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def process_cloth_image(net, image_path, output_path, device='cuda'):
    """
    Process a single cloth image to generate mask
    
    Args:
        net: U-2-Net model
        image_path: Path to input cloth image
        output_path: Path to save output mask
        device: 'cuda' or 'cpu'
    """
    print(f"[INFO] Processing: {Path(image_path).name}")
    
    # Read image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    if device == 'cuda' and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    # Run inference
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_tensor)
    
    # Get prediction (use d1, the final output)
    pred = d1[:, 0, :, :]
    pred = normalize_output(pred)
    
    # Convert to numpy
    pred = pred.squeeze().cpu().numpy()
    
    # Convert to PIL Image
    pred = (pred * 255).astype(np.uint8)
    mask = Image.fromarray(pred)
    
    # Resize to original image size
    mask = mask.resize(original_size, Image.BILINEAR)
    
    # Apply threshold to get binary mask
    mask_np = np.array(mask)
    mask_np = (mask_np > 128).astype(np.uint8) * 255
    
    # Save mask
    mask_final = Image.fromarray(mask_np)
    mask_final.save(output_path)
    
    print(f"[SUCCESS] Saved mask: {Path(output_path).name}")
    return True


def process_directory(net, input_dir, output_dir, device='cuda'):
    """Process all cloth images in directory"""
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
        print(f"[ERROR] No images found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(image_files)} cloth images to process")
    print("=" * 60)
    
    success_count = 0
    for img_file in image_files:
        # Output filename (keep same name, change to .png)
        output_file = output_path / img_file.with_suffix('.png').name
        
        try:
            if process_cloth_image(net, img_file, output_file, device):
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_file.name}: {e}")
        
        print("-" * 60)
    
    print("=" * 60)
    print(f"[COMPLETE] Successfully processed {success_count}/{len(image_files)} images")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloth masks using U-2-Net"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing cloth images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for cloth masks')
    parser.add_argument('--model_path', type=str,
                        default='../cloth_segmentation/saved_models/u2net.pth',
                        help='Path to U-2-Net model checkpoint')
    parser.add_argument('--model_type', type=str, default='u2net',
                        choices=['u2net', 'u2netp'],
                        help='Model type: u2net or u2netp (smaller)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup model path
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    
    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}")
        print("\nPlease download the model:")
        print("1. Visit: https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ")
        print("2. Download u2net.pth")
        print(f"3. Save to: {model_path}")
        print("\nOr run: python scripts/download_u2net_checkpoint.py")
        return
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("Cloth Mask Generation (U-2-Net)")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    net = load_model(str(model_path), args.model_type, device)
    
    # Process images
    process_directory(net, args.input_dir, args.output_dir, device)


if __name__ == '__main__':
    main()
