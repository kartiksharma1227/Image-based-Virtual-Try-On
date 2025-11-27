"""
Download checkpoint for Lightweight Human Pose Estimation
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading: {url}")
    print(f"To: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Download complete!")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "checkpoint_iter_370000.pth"
    
    if checkpoint_path.exists():
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Checkpoint URL (from the original repository)
    checkpoint_url = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"
    
    try:
        download_file(checkpoint_url, checkpoint_path)
        
        # Verify file size (should be around 13MB)
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"\nFile size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 10:
            print("⚠ Warning: File size seems too small. Download may have failed.")
        else:
            print(f"✓ Checkpoint downloaded successfully!")
            print(f"Location: {checkpoint_path}")
        
    except Exception as e:
        print(f"\n✗ Error downloading checkpoint: {e}")
        print("\nAlternative download method:")
        print(f"1. Visit: {checkpoint_url}")
        print(f"2. Download the file manually")
        print(f"3. Save it to: {checkpoint_path}")


if __name__ == '__main__':
    main()
