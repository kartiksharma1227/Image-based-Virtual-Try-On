"""
OpenPose Preprocessing Script for StableVITON
Uses Lightweight Human Pose Estimation

This script processes person images and generates:
1. OpenPose JSON keypoints (openpose_json/)
2. Visualized pose images (openpose_img/)
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
import math
from pathlib import Path
import argparse

# Add openpose_lightweight to path
SCRIPT_DIR = Path(__file__).parent
OPENPOSE_DIR = SCRIPT_DIR.parent / "openpose_lightweight"
sys.path.insert(0, str(OPENPOSE_DIR))

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses


def normalize(img, img_mean, img_scale):
    """Normalize image"""
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    """Pad image to meet minimum dimensions and stride requirements"""
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                     cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


class OpenPoseProcessor:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize OpenPose Processor
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"[INFO] Loading model on {device}...")
        
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_state(self.net, checkpoint)
        
        self.net = self.net.eval()
        if device == 'cuda':
            self.net = self.net.cuda()
        
        print("[INFO] Model loaded successfully!")
    
    def preprocess_image(self, img, height_size=256, stride=8):
        """Preprocess image for inference"""
        height, width, _ = img.shape
        scale = height_size / height
        
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, np.array([128, 128, 128], np.float32), np.float32(1/256))
        min_dims = [height_size, max(scaled_img.shape[1], height_size)]
        padded_img, pad = pad_width(scaled_img, stride, (0, 0, 0), min_dims)
        
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if self.device == 'cuda':
            tensor_img = tensor_img.cuda()
        
        return tensor_img, scale, pad
    
    def infer(self, img, height_size=256, stride=8, upsample_ratio=4):
        """
        Run pose estimation inference
        
        Args:
            img: Input image (BGR format)
            height_size: Height for inference
            stride: Network stride
            upsample_ratio: Upsampling ratio for heatmaps
            
        Returns:
            pose_entries, all_keypoints: Detected poses and keypoints
        """
        tensor_img, scale, pad = self.preprocess_image(img, height_size, stride)
        
        with torch.no_grad():
            stages_output = self.net(tensor_img)
        
        heatmaps = stages_output[-2]
        pafs = stages_output[-1]
        
        heatmaps = heatmaps.cpu().numpy()[0]
        pafs = pafs.cpu().numpy()[0]
        
        # Resize heatmaps
        heatmaps = np.transpose(heatmaps, (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
        
        # Resize pafs
        pafs = np.transpose(pafs, (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
        
        # Extract keypoints for each type
        num_keypoints = 18
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        
        # Rescale keypoints to original image size
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        return pose_entries, all_keypoints
    
    def convert_to_openpose_format(self, pose_entries, all_keypoints):
        """
        Convert keypoints to OpenPose JSON format
        
        Returns:
            dict: OpenPose format JSON data
        """
        people = []
        
        for pose_entry in pose_entries:
            if pose_entry.shape[0] == 0:
                continue
            
            keypoints_2d = []
            
            # OpenPose has 18 keypoints, lightweight has 18 as well
            # Format: [x1, y1, c1, x2, y2, c2, ...]
            for kpt_idx in range(18):
                if pose_entry[kpt_idx] != -1:
                    kpt_id = int(pose_entry[kpt_idx])
                    x, y, conf = all_keypoints[kpt_id, 0:3]
                    keypoints_2d.extend([float(x), float(y), float(conf)])
                else:
                    keypoints_2d.extend([0.0, 0.0, 0.0])
            
            person = {
                "pose_keypoints_2d": keypoints_2d,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            people.append(person)
        
        return {"version": 1.3, "people": people}
    
    def visualize_poses(self, img, pose_entries, all_keypoints):
        """Draw poses on image"""
        img_vis = img.copy()
        
        for pose_entry in pose_entries:
            if len(pose_entry) == 0:
                continue
            
            pose = Pose(pose_entry, all_keypoints)
            pose.draw(img_vis)
        
        return img_vis
    
    def process_image(self, img_path, output_json_dir, output_img_dir, height_size=256):
        """
        Process a single image
        
        Args:
            img_path: Path to input image
            output_json_dir: Directory to save JSON
            output_img_dir: Directory to save visualization
            height_size: Height for inference
        """
        img_name = Path(img_path).name
        base_name = Path(img_path).stem
        
        print(f"[INFO] Processing: {img_name}")
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Failed to read image: {img_path}")
            return False
        
        # Run inference
        pose_entries, all_keypoints = self.infer(img, height_size)
        
        if len(pose_entries) == 0:
            print(f"[WARNING] No pose detected in: {img_name}")
        
        # Save JSON
        json_data = self.convert_to_openpose_format(pose_entries, all_keypoints)
        json_path = os.path.join(output_json_dir, f"{base_name}_keypoints.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save visualization
        img_vis = self.visualize_poses(img, pose_entries, all_keypoints)
        img_out_path = os.path.join(output_img_dir, f"{base_name}_rendered.png")
        cv2.imwrite(img_out_path, img_vis)
        
        print(f"[SUCCESS] Saved: {json_path}")
        print(f"[SUCCESS] Saved: {img_out_path}")
        
        return True
    
    def process_directory(self, input_dir, output_json_dir, output_img_dir, height_size=256):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_json_dir: Directory to save JSON files
            output_img_dir: Directory to save visualizations
            height_size: Height for inference
        """
        os.makedirs(output_json_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            print(f"[ERROR] No images found in: {input_dir}")
            return
        
        print(f"[INFO] Found {len(image_files)} images to process")
        print("=" * 60)
        
        success_count = 0
        for img_path in image_files:
            if self.process_image(img_path, output_json_dir, output_img_dir, height_size):
                success_count += 1
            print("-" * 60)
        
        print("=" * 60)
        print(f"[COMPLETE] Successfully processed {success_count}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description="OpenPose preprocessing for StableVITON")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing person images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory (will create openpose_json/ and openpose_img/ subdirs)')
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/checkpoint_iter_370000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--height_size', type=int, default=256,
                        help='Height size for inference (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nPlease download the checkpoint first:")
        print("Run: python scripts/download_checkpoint.py")
        return
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create output directories
    output_json_dir = os.path.join(args.output_dir, 'openpose_json')
    output_img_dir = os.path.join(args.output_dir, 'openpose_img')
    
    # Initialize processor
    processor = OpenPoseProcessor(str(checkpoint_path), device=device)
    
    # Process images
    processor.process_directory(
        args.input_dir,
        output_json_dir,
        output_img_dir,
        height_size=args.height_size
    )


if __name__ == '__main__':
    main()
