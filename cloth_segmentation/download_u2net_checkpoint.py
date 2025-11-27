"""
Download U-2-Net checkpoint for cloth segmentation
"""

import os
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
    print(f"Downloading from: {url}")
    print(f"To: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Download complete!")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    cloth_seg_dir = script_dir.parent / "cloth_segmentation"
    saved_models_dir = cloth_seg_dir / "saved_models"
    saved_models_dir.mkdir(exist_ok=True)
    
    model_path = saved_models_dir / "u2net.pth"
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print("=" * 60)
    print("Downloading U-2-Net Model for Cloth Segmentation")
    print("=" * 60)
    
    # Direct download URL from Google Drive
    # Note: Google Drive direct download links can be tricky
    model_url = "https://drive.google.com/uc?export=download&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
    
    try:
        download_file(model_url, model_path)
        
        # Verify file size (should be around 176MB)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\nFile size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 100:
            print("⚠ Warning: File size seems too small. Download may have failed.")
            print("\nAlternative download methods:")
            print("1. Visit: https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view")
            print("2. Download manually")
            print(f"3. Save to: {model_path}")
        else:
            print(f"✓ Model downloaded successfully!")
            print(f"Location: {model_path}")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nManual download instructions:")
        print("=" * 60)
        print("1. Visit: https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view")
        print("2. Click 'Download' button")
        print(f"3. Save the file as: {model_path}")
        print("=" * 60)
        print("\nOr use gdown:")
        print(f"  pip install gdown")
        print(f"  gdown 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O {model_path}")


if __name__ == '__main__':
    main()
