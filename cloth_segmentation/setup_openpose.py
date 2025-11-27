"""
Setup script for OpenPose preprocessing module
Installs dependencies and downloads checkpoint
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed!")
        return False
    
    print(f"\n✓ {description} completed successfully!")
    return True


def main():
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    openpose_dir = root_dir / "openpose_lightweight"
    
    print("="*60)
    print("OpenPose Preprocessing Setup")
    print("="*60)
    
    # Step 1: Install openpose_lightweight requirements
    openpose_req = openpose_dir / "requirements.txt"
    if openpose_req.exists():
        if not run_command(
            f'pip install -r "{openpose_req}"',
            "Installing OpenPose Lightweight dependencies"
        ):
            return
    else:
        print(f"⚠ Warning: {openpose_req} not found, skipping...")
    
    # Step 2: Install additional requirements
    toolkit_req = root_dir / "requirements_openpose.txt"
    if not run_command(
        f'pip install -r "{toolkit_req}"',
        "Installing additional dependencies"
    ):
        return
    
    # Step 3: Download checkpoint
    download_script = script_dir / "download_checkpoint.py"
    if not run_command(
        f'python "{download_script}"',
        "Downloading model checkpoint"
    ):
        return
    
    print("\n" + "="*60)
    print("✓ OpenPose preprocessing setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your person images in: input_images/person/")
    print("2. Run: python scripts/run_openpose.py --input_dir input_images/person --output_dir output")
    print("="*60)


if __name__ == '__main__':
    main()
