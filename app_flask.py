"""
Flask Web Application for StableVITON Virtual Try-On
Beautiful and simple UI for virtual garment try-on
"""
import os
import sys
import subprocess
import time
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image
import threading
import shutil
from logging_config import setup_app_logger, log_exception, log_gpu_status

# Initialize logger
logger = setup_app_logger()
logger.info("="*70)
logger.info("Flask Application Starting...")
logger.info("="*70)

app = Flask(__name__)
app.secret_key = 'stableviton_secret_key_2024'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Base directory
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER_IMAGES = BASE_DIR / 'Test-Images'
UPLOAD_FOLDER_CLOTHES = BASE_DIR / 'Test-Clothes'
OUTPUT_FOLDER = BASE_DIR / 'Without_repaint_output' / 'unpair'

# Ensure directories exist
UPLOAD_FOLDER_IMAGES.mkdir(exist_ok=True)
UPLOAD_FOLDER_CLOTHES.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Directory to store original dimensions
DIMENSIONS_FOLDER = BASE_DIR / 'original_dimensions'
DIMENSIONS_FOLDER.mkdir(exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Global progress tracking
progress_data = {}


def clear_gpu_cache():
    """Clear GPU memory cache to prevent OOM errors"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.debug("Clearing GPU cache...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("GPU cache cleared successfully")
            log_gpu_status(logger)
        else:
            logger.debug("No GPU available to clear")
    except ImportError:
        logger.debug("PyTorch not available in this environment - skipping GPU cache clear")
    except Exception as e:
        log_exception(logger, e, "Error clearing GPU cache")


def get_conda_path():
    """Get the path to conda executable"""
    logger.debug("Searching for conda executable...")
    # Try to find conda in common locations
    conda_paths = [
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        r"C:\Users\{}\anaconda3\Scripts\conda.exe".format(os.environ.get('USERNAME', '')),
        r"C:\Users\{}\miniconda3\Scripts\conda.exe".format(os.environ.get('USERNAME', '')),
    ]
    
    # Check if conda is in PATH
    conda_from_path = shutil.which('conda')
    if conda_from_path:
        logger.info(f"Found conda in PATH: {conda_from_path}")
        return conda_from_path
    
    # Check common locations
    for path in conda_paths:
        if os.path.exists(path):
            logger.info(f"Found conda at: {path}")
            return path
    
    # Last resort - try using conda.bat
    logger.warning("Using 'conda' command (not found in common locations)")
    return 'conda'


def run_conda_command(env_name, script_path, args):
    """
    Run a Python script in a specific conda environment
    
    Args:
        env_name: Name of the conda environment
        script_path: Path to the Python script
        args: List of arguments for the script
    
    Returns:
        subprocess.CompletedProcess object
    """
    logger.info(f"Running script in conda environment: {env_name}")
    logger.debug(f"Script: {script_path}")
    logger.debug(f"Args: {args}")
    
    conda_exe = get_conda_path()
    
    # Build the command for Windows
    # Use conda run with --no-capture-output for real-time output
    cmd = [
        conda_exe, 'run', '-n', env_name, '--no-capture-output',
        'python', str(script_path)
    ] + args
    
    logger.debug(f"Full command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        check=True,
        capture_output=True,
        text=True,
        shell=False
    )
    
    if result.stdout:
        logger.debug(f"Command stdout: {result.stdout[:500]}...")  # First 500 chars
    if result.stderr:
        logger.warning(f"Command stderr: {result.stderr[:500]}...")  # First 500 chars
    
    logger.info(f"Script completed successfully in {env_name}")
    return result


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def store_original_dimensions(image_path, name):
    """Store original image dimensions before resizing"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Store dimensions in a text file
        dim_file = DIMENSIONS_FOLDER / f"{name}_dimensions.txt"
        with open(dim_file, 'w') as f:
            f.write(f"{width},{height}")
        
        logger.info(f"Stored original dimensions for {name}: {width}x{height}")
        return width, height
    except Exception as e:
        logger.error(f"Failed to store dimensions for {name}: {e}")
        return None, None


def get_original_dimensions(name):
    """Retrieve stored original dimensions"""
    try:
        dim_file = DIMENSIONS_FOLDER / f"{name}_dimensions.txt"
        if dim_file.exists():
            with open(dim_file, 'r') as f:
                width, height = map(int, f.read().strip().split(','))
            logger.debug(f"Retrieved dimensions for {name}: {width}x{height}")
            return width, height
    except Exception as e:
        logger.error(f"Failed to retrieve dimensions for {name}: {e}")
    return None, None


def resize_output_to_original(output_path, person_name):
    """Resize output image to original person image dimensions"""
    try:
        # Get original dimensions
        orig_width, orig_height = get_original_dimensions(person_name)
        
        if orig_width is None or orig_height is None:
            logger.warning(f"Could not retrieve original dimensions for {person_name}, skipping resize")
            return False
        
        # Load output image
        if not output_path.exists():
            logger.error(f"Output image not found: {output_path}")
            return False
        
        logger.info(f"Resizing output from 768x1024 to original {orig_width}x{orig_height}")
        
        img = Image.open(output_path)
        
        # Resize to original dimensions
        resized_img = img.resize((orig_width, orig_height), Image.LANCZOS)
        
        # Save resized image
        resized_img.save(output_path, 'JPEG', quality=95)
        
        logger.info(f"Output image resized successfully: {output_path}")
        return True
        
    except Exception as e:
        log_exception(logger, e, f"Failed to resize output image: {output_path}")
        return False


def convert_to_jpg(image_path):
    """Convert image to JPG format if needed"""
    logger.debug(f"Converting image to JPG: {image_path}")
    try:
        img = Image.open(image_path)
        original_mode = img.mode
        logger.debug(f"Original image mode: {original_mode}")
        
        if img.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
            logger.debug(f"Converted from {original_mode} to RGB")
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            logger.debug(f"Converted from {original_mode} to RGB")
        
        # Save as JPG
        img.save(image_path, 'JPEG', quality=95)
        logger.info(f"Image converted and saved: {image_path}")
        return True
    except Exception as e:
        log_exception(logger, e, f"Error converting image: {image_path}")
        return False


def update_progress(session_id, stage, progress, message):
    """Update progress for a session"""
    logger.info(f"[Session {session_id[:8]}] Progress: {stage} - {progress}% - {message}")
    progress_data[session_id] = {
        'stage': stage,
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }


def run_preprocessing(person_name, cloth_name, session_id):
    """Run preprocessing pipeline"""
    logger.info(f"Starting preprocessing for person={person_name}, cloth={cloth_name}")
    try:
        update_progress(session_id, 'preprocessing', 10, 'Starting preprocessing...')
        
        preprocess_script = BASE_DIR / 'preprocessing_toolkit' / 'completepreprocessing.py'
        
        if not preprocess_script.exists():
            error_msg = f'Preprocessing script not found: {preprocess_script}'
            logger.error(error_msg)
            update_progress(session_id, 'error', 0, 'Preprocessing script not found')
            return False
        
        logger.info(f"Found preprocessing script: {preprocess_script}")
        update_progress(session_id, 'preprocessing', 20, 'Extracting human parsing...')
        
        # Use the helper function to run in viton-pip environment
        args = [
            '--person', person_name,
            '--cloth', cloth_name,
            '--sequential'
        ]
        
        logger.info("Executing preprocessing script...")
        result = run_conda_command('viton-pip', preprocess_script, args)
        
        logger.info("Preprocessing completed successfully")
        update_progress(session_id, 'preprocessing', 50, 'Preprocessing completed!')
        
        # Clear GPU cache after preprocessing
        logger.info("Clearing GPU cache after preprocessing...")
        clear_gpu_cache()
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f'Preprocessing failed: {e.stderr if e.stderr else str(e)}'
        log_exception(logger, e, "Preprocessing subprocess error")
        update_progress(session_id, 'error', 0, error_msg)
        logger.error(f"stdout: {e.stdout if e.stdout else 'None'}")
        return False
    except Exception as e:
        log_exception(logger, e, "Preprocessing exception")
        update_progress(session_id, 'error', 0, f'Error: {str(e)}')
        return False


def create_test_pairs_file(person_name, cloth_name):
    """Create test_pairs.txt file"""
    logger.info(f"Creating test_pairs.txt for {person_name} and {cloth_name}")
    test_pairs_file = BASE_DIR / 'my_data' / 'test_pairs.txt'
    test_pairs_file.parent.mkdir(exist_ok=True)
    
    pair_entry = f"{person_name}.jpg {cloth_name}.jpg"
    
    with open(test_pairs_file, 'w') as f:
        f.write(pair_entry)
    
    logger.info(f"Created test_pairs.txt: {pair_entry}")


def run_inference(person_name, cloth_name, session_id):
    """Run inference"""
    logger.info(f"Starting inference for person={person_name}, cloth={cloth_name}")
    try:
        update_progress(session_id, 'inference', 60, 'Initializing model...')
        
        inference_script = BASE_DIR / 'inference.py'
        config_file = BASE_DIR / 'configs' / 'VITONHD.yaml'
        model_path = BASE_DIR / 'ckpts' / 'VITONHD.ckpt'
        data_root = BASE_DIR / 'my_data'
        output_dir = BASE_DIR / 'Without_repaint_output'
        
        # Verify files exist
        logger.debug("Verifying required files...")
        if not inference_script.exists():
            logger.error(f"Inference script not found: {inference_script}")
            update_progress(session_id, 'error', 0, 'Inference script missing')
            return False
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            update_progress(session_id, 'error', 0, 'Config file missing')
            return False
        if not model_path.exists():
            logger.error(f"Model checkpoint not found: {model_path}")
            update_progress(session_id, 'error', 0, 'Model checkpoint missing')
            return False
        
        logger.info("All required files found")
        update_progress(session_id, 'inference', 70, 'Running virtual try-on...')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logger.debug("Set CUDA_VISIBLE_DEVICES=0")
        
        # Use the helper function to run in viton-pip environment
        args = [
            '--config_path', str(config_file),
            '--batch_size', '1',
            '--model_load_path', str(model_path),
            '--data_root_dir', str(data_root),
            '--unpair',
            '--save_dir', str(output_dir)
        ]
        
        logger.info("Executing inference script...")
        log_gpu_status(logger)
        
        result = run_conda_command('viton-pip', inference_script, args)
        
        logger.info("Inference script completed")
        update_progress(session_id, 'inference', 90, 'Generating final output...')
        
        # Verify output exists
        output_file = output_dir / 'unpair' / f"{person_name}_{cloth_name}.jpg"
        logger.debug(f"Checking for output file: {output_file}")
        
        if output_file.exists():
            logger.info(f"Output file generated successfully: {output_file}")
            
            # Resize output to original dimensions
            update_progress(session_id, 'inference', 95, 'Resizing to original dimensions...')
            resize_success = resize_output_to_original(output_file, person_name)
            
            if resize_success:
                logger.info("Output resized to original dimensions")
            else:
                logger.warning("Output resize skipped or failed, keeping 768x1024")
            
            update_progress(session_id, 'complete', 100, 'Virtual try-on complete!')
            
            # Clear GPU cache after inference
            logger.info("Clearing GPU cache after inference...")
            clear_gpu_cache()
            
            return True
        else:
            logger.error(f"Output file not generated: {output_file}")
            update_progress(session_id, 'error', 0, 'Output file not generated')
            return False
        
    except subprocess.CalledProcessError as e:
        error_msg = f'Inference failed: {e.stderr if e.stderr else str(e)}'
        log_exception(logger, e, "Inference subprocess error")
        update_progress(session_id, 'error', 0, error_msg)
        logger.error(f"stdout: {e.stdout if e.stdout else 'None'}")
        return False
    except Exception as e:
        log_exception(logger, e, "Inference exception")
        update_progress(session_id, 'error', 0, f'Error: {str(e)}')
        return False


def process_virtual_tryon(person_name, cloth_name, session_id):
    """Complete virtual try-on process (runs in background thread)"""
    logger.info(f"="*70)
    logger.info(f"Starting complete virtual try-on process")
    logger.info(f"Person: {person_name}, Cloth: {cloth_name}, Session: {session_id[:8]}")
    logger.info(f"="*70)
    
    try:
        # Create test pairs file
        logger.info("Step 1: Creating test pairs file")
        create_test_pairs_file(person_name, cloth_name)
        
        # Run preprocessing
        logger.info("Step 2: Running preprocessing")
        if not run_preprocessing(person_name, cloth_name, session_id):
            logger.error("Preprocessing failed - aborting")
            return
        
        logger.info("Preprocessing completed successfully")
        
        # Run inference
        logger.info("Step 3: Running inference")
        if not run_inference(person_name, cloth_name, session_id):
            logger.error("Inference failed - aborting")
            return
        
        logger.info("Inference completed successfully")
        logger.info("="*70)
        logger.info("Virtual try-on process completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        log_exception(logger, e, "Fatal error in process_virtual_tryon")
        update_progress(session_id, 'error', 0, f'Process failed: {str(e)}')


@app.route('/')
def index():
    """Render main page"""
    # Generate unique session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        logger.info(f"New session created: {session['session_id'][:8]}")
    else:
        logger.debug(f"Existing session: {session['session_id'][:8]}")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    logger.info("Upload request received")
    try:
        # Check if files are present
        if 'person_image' not in request.files or 'cloth_image' not in request.files:
            logger.warning("Upload failed: Both images are required")
            return jsonify({'success': False, 'error': 'Both images are required'})
        
        person_file = request.files['person_image']
        cloth_file = request.files['cloth_image']
        
        logger.debug(f"Person file: {person_file.filename}")
        logger.debug(f"Cloth file: {cloth_file.filename}")
        
        # Validate files
        if person_file.filename == '' or cloth_file.filename == '':
            logger.warning("Upload failed: No files selected")
            return jsonify({'success': False, 'error': 'No files selected'})
        
        if not (allowed_file(person_file.filename) and allowed_file(cloth_file.filename)):
            logger.warning(f"Upload failed: Invalid file types - {person_file.filename}, {cloth_file.filename}")
            return jsonify({'success': False, 'error': 'Only JPG/JPEG images are allowed'})
        
        # Generate unique filenames with _00 suffix
        timestamp = str(int(time.time()))
        person_name = f"person_{timestamp}_00"
        cloth_name = f"cloth_{timestamp}_00"
        
        person_path = UPLOAD_FOLDER_IMAGES / f"{person_name}.jpg"
        cloth_path = UPLOAD_FOLDER_CLOTHES / f"{cloth_name}.jpg"
        
        logger.info(f"Saving files: {person_name}.jpg, {cloth_name}.jpg")
        
        # Save files
        person_file.save(str(person_path))
        cloth_file.save(str(cloth_path))
        
        logger.debug(f"Person file saved to: {person_path}")
        logger.debug(f"Cloth file saved to: {cloth_path}")
        
        # Store original dimensions before any processing
        store_original_dimensions(person_path, person_name)
        store_original_dimensions(cloth_path, cloth_name)
        
        # Convert to JPG if needed
        if not convert_to_jpg(person_path):
            logger.error(f"Failed to convert person image: {person_path}")
            return jsonify({'success': False, 'error': 'Failed to process person image'})
        
        if not convert_to_jpg(cloth_path):
            logger.error(f"Failed to convert cloth image: {cloth_path}")
            return jsonify({'success': False, 'error': 'Failed to process cloth image'})
        
        logger.info(f"Upload successful: {person_name}, {cloth_name}")
        
        return jsonify({
            'success': True,
            'person_name': person_name,
            'cloth_name': cloth_name
        })
        
    except Exception as e:
        log_exception(logger, e, "Upload error")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/process', methods=['POST'])
def process():
    """Start the virtual try-on process"""
    logger.info("Process request received")
    try:
        data = request.get_json()
        person_name = data.get('person_name')
        cloth_name = data.get('cloth_name')
        
        logger.debug(f"Process parameters: person={person_name}, cloth={cloth_name}")
        
        if not person_name or not cloth_name:
            logger.warning("Process failed: Missing parameters")
            return jsonify({'success': False, 'error': 'Missing parameters'})
        
        # Get or create session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        logger.info(f"Starting processing for session {session_id[:8]}")
        
        # Initialize progress
        update_progress(session_id, 'starting', 0, 'Starting virtual try-on...')
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_virtual_tryon,
            args=(person_name, cloth_name, session_id)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Background thread started for session {session_id[:8]}")
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
        
    except Exception as e:
        log_exception(logger, e, "Process error")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/progress')
def get_progress():
    """Get processing progress"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in progress_data:
        return jsonify({
            'stage': 'idle',
            'progress': 0,
            'message': 'Waiting...'
        })
    
    return jsonify(progress_data[session_id])


@app.route('/uploads/person/<filename>')
def uploaded_person(filename):
    """Serve uploaded person images"""
    return send_from_directory(UPLOAD_FOLDER_IMAGES, filename)


@app.route('/uploads/cloth/<filename>')
def uploaded_cloth(filename):
    """Serve uploaded cloth images"""
    return send_from_directory(UPLOAD_FOLDER_CLOTHES, filename)


@app.route('/output/<filename>')
def output_image(filename):
    """Serve output images"""
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  🎨 STABLEVITON WEB APPLICATION")
    print("=" * 70)
    print("  Starting Flask server...")
    print("  Open your browser and navigate to: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
