"""
Test script for OpenPose preprocessing
Verifies that everything is set up correctly
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'tqdm',
        'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print("Run: python scripts/setup_openpose.py")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_checkpoint():
    """Check if model checkpoint exists"""
    print("\n" + "="*60)
    print("Checking Model Checkpoint")
    print("="*60)
    
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir.parent / "checkpoints" / "checkpoint_iter_370000.pth"
    
    if checkpoint_path.exists():
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Checkpoint found: {checkpoint_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Checkpoint NOT found: {checkpoint_path}")
        print("Run: python scripts/download_checkpoint.py")
        return False


def check_openpose_repo():
    """Check if openpose_lightweight repository exists"""
    print("\n" + "="*60)
    print("Checking OpenPose Repository")
    print("="*60)
    
    script_dir = Path(__file__).parent
    openpose_dir = script_dir.parent / "openpose_lightweight"
    
    if openpose_dir.exists():
        print(f"✓ Repository found: {openpose_dir}")
        
        # Check for key files
        key_files = [
            "models/with_mobilenet.py",
            "modules/keypoints.py",
            "modules/pose.py"
        ]
        
        all_found = True
        for file in key_files:
            file_path = openpose_dir / file
            if file_path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} - NOT FOUND")
                all_found = False
        
        return all_found
    else:
        print(f"✗ Repository NOT found: {openpose_dir}")
        print("The repository should have been cloned during setup.")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("Checking CUDA")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA NOT available")
            print("  Processing will use CPU (slower)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\n" + "="*60)
    print("Checking Directories")
    print("="*60)
    
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    required_dirs = [
        "checkpoints",
        "input_images/person",
        "input_images/cloth",
        "output/openpose_json",
        "output/openpose_img"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = root_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    print("="*60)
    print("OpenPose Preprocessing - Environment Test")
    print("="*60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Checkpoint": check_checkpoint(),
        "OpenPose Repo": check_openpose_repo(),
        "CUDA": check_cuda(),
        "Directories": check_directories()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_critical_pass = (
        results["Dependencies"] and 
        results["Checkpoint"] and 
        results["OpenPose Repo"] and
        results["Directories"]
    )
    
    print("\n" + "="*60)
    if all_critical_pass:
        print("✓ Environment is ready!")
        print("="*60)
        print("\nYou can now run:")
        print("python scripts/run_openpose.py --input_dir input_images/person --output_dir output")
    else:
        print("✗ Environment is NOT ready")
        print("="*60)
        print("\nPlease fix the issues above.")
        print("Run: python scripts/setup_openpose.py")
    
    print("="*60)


if __name__ == '__main__':
    main()
