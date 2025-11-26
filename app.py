"""
Complete Virtual Try-On Application for StableVITON
This script integrates preprocessing and inference into a single workflow.

Usage: python app.py --person PERSON_NAME --cloth CLOTH_NAME
Example: python app.py --person 00190_00 --cloth 00158_00
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from PIL import Image

def run_complete_preprocessing(person_name, cloth_name, base_dir):
    """
    Run the complete preprocessing pipeline
    """
    print("\n" + "=" * 70)
    print("  STEP 1: PREPROCESSING")
    print("=" * 70)
    
    preprocess_script = base_dir / 'preprocessing_toolkit' / 'completepreprocessing.py'
    
    if not preprocess_script.exists():
        print(f"✗ ERROR: Preprocessing script not found: {preprocess_script}")
        sys.exit(1)
    
    # Run preprocessing
    cmd = [
        sys.executable,
        str(preprocess_script),
        '--person', person_name,
        '--cloth', cloth_name,
        '--sequential'  # Use sequential mode for more stable execution
    ]
    
    print(f"Running preprocessing...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(base_dir),
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Preprocessing completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Preprocessing failed with exit code {e.returncode}")
        return False


def create_test_pairs_file(person_name, cloth_name, base_dir):
    """
    Create test_pairs.txt with single entry (overwrite mode)
    """
    test_pairs_file = base_dir / 'my_data' / 'test_pairs.txt'
    pair_entry = f"{person_name}.jpg {cloth_name}.jpg"
    
    # Write (overwrite, not append) - single line, no trailing newline
    with open(test_pairs_file, 'w') as f:
        f.write(pair_entry)
    
    print(f"✓ Created test_pairs.txt: {pair_entry}")


def run_inference(base_dir):
    """
    Run the inference using the StableVITON model
    """
    print("\n" + "=" * 70)
    print("  STEP 2: INFERENCE (Virtual Try-On)")
    print("=" * 70)
    
    inference_script = base_dir / 'inference.py'
    config_file = base_dir / 'configs' / 'VITONHD.yaml'
    model_path = base_dir / 'ckpts' / 'VITONHD.ckpt'
    data_root = base_dir / 'my_data'
    output_dir = base_dir / 'Without_repaint_output'
    
    # Verify required files exist
    if not inference_script.exists():
        print(f"✗ ERROR: Inference script not found: {inference_script}")
        sys.exit(1)
    
    if not config_file.exists():
        print(f"✗ ERROR: Config file not found: {config_file}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"✗ ERROR: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Build inference command
    # We'll use conda run to ensure we're in the viton-pip environment
    cmd = [
        'conda', 'run', '-n', 'viton-pip', '--no-capture-output',
        'python', str(inference_script),
        '--config_path', str(config_file),
        '--batch_size', '1',
        '--model_load_path', str(model_path),
        '--data_root_dir', str(data_root),
        '--unpair',
        '--save_dir', str(output_dir)
    ]
    
    print(f"Running inference...")
    print(f"Output will be saved to: {output_dir / 'unpair'}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(base_dir),
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Inference completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Inference failed with exit code {e.returncode}")
        return False


def display_results(person_name, cloth_name, base_dir):
    """
    Display information about the generated results
    """
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    output_dir = base_dir / 'Without_repaint_output' / 'unpair'
    expected_output = output_dir / f"{person_name}_{cloth_name}.jpg"
    
    if expected_output.exists():
        size = expected_output.stat().st_size / 1024  # KB
        
        # Try to get image dimensions
        try:
            img = Image.open(expected_output)
            width, height = img.size
            print(f"✓ Virtual Try-On Result:")
            print(f"  File: {expected_output.name}")
            print(f"  Path: {expected_output}")
            print(f"  Size: {size:.1f} KB")
            print(f"  Dimensions: {width}x{height}")
        except Exception as e:
            print(f"✓ Output saved: {expected_output}")
            print(f"  Size: {size:.1f} KB")
    else:
        print(f"⚠ Warning: Expected output not found: {expected_output}")
        print(f"\nListing all files in {output_dir}:")
        if output_dir.exists():
            for file in output_dir.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
        else:
            print(f"  Output directory doesn't exist")


def main():
    """
    Main application function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Complete Virtual Try-On Application for StableVITON'
    )
    parser.add_argument('--person', type=str, required=True,
                       help='Person image name (without .jpg extension)')
    parser.add_argument('--cloth', type=str, required=True,
                       help='Cloth image name (without .jpg extension)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing (use existing preprocessed data)')
    
    args = parser.parse_args()
    
    person_name = args.person
    cloth_name = args.cloth
    
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Print header
    print("\n" + "=" * 70)
    print("  STABLEVITON - COMPLETE VIRTUAL TRY-ON APPLICATION")
    print("=" * 70)
    print(f"Person: {person_name}")
    print(f"Cloth: {cloth_name}")
    print(f"Base Directory: {base_dir}")
    print("=" * 70)
    
    # Verify source images exist
    if not args.skip_preprocessing:
        test_images = base_dir / 'Test-Images'
        test_clothes = base_dir / 'Test-Clothes'
        
        person_src = test_images / f'{person_name}.jpg'
        cloth_src = test_clothes / f'{cloth_name}.jpg'
        
        if not person_src.exists():
            print(f"\n✗ ERROR: Person image not found: {person_src}")
            sys.exit(1)
        
        if not cloth_src.exists():
            print(f"\n✗ ERROR: Cloth image not found: {cloth_src}")
            sys.exit(1)
        
        print(f"\n✓ Source images verified")
    
    # Start timer
    total_start_time = time.time()
    
    # Step 1: Run preprocessing (unless skipped)
    if not args.skip_preprocessing:
        preprocessing_success = run_complete_preprocessing(person_name, cloth_name, base_dir)
        
        if not preprocessing_success:
            print("\n✗ FATAL ERROR: Preprocessing failed. Cannot proceed to inference.")
            sys.exit(1)
    else:
        print("\n⚠ Skipping preprocessing (using existing data)")
    
    # Create test_pairs.txt file (overwrite mode, single line)
    create_test_pairs_file(person_name, cloth_name, base_dir)
    
    # Step 2: Run inference
    inference_success = run_inference(base_dir)
    
    if not inference_success:
        print("\n✗ FATAL ERROR: Inference failed.")
        sys.exit(1)
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    
    # Display results
    display_results(person_name, cloth_name, base_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("  🎉 SUCCESS! VIRTUAL TRY-ON COMPLETE!")
    print("=" * 70)
    print(f"Total Time: {total_elapsed:.1f} seconds")
    print(f"\nYour virtual try-on result is ready!")
    print(f"Output: Without_repaint_output/unpair/{person_name}_{cloth_name}.jpg")
    print("=" * 70)


if __name__ == "__main__":
    main()
