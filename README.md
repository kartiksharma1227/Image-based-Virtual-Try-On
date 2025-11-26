# StableVITON - Virtual Try-On System

A complete deep learning-based virtual try-on system that allows you to visualize how clothes would look on a person using AI. This project includes preprocessing pipelines, model training capabilities, and a beautiful web interface.

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Model Checkpoints](#model-checkpoints)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Web Application](#web-application)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Complete Preprocessing Pipeline**: Automated human parsing, pose detection, and garment segmentation
- **Model Training**: Full training pipeline with validation and checkpointing
- **Web Interface**: Beautiful Flask-based UI with drag-and-drop upload
- **Real-time Progress Tracking**: Live updates during preprocessing and inference
- **Multi-threaded Processing**: Background processing for responsive UI
- **Automatic Format Conversion**: Handles various image formats automatically
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 🖥️ System Requirements

### Hardware

- **GPU**: NVIDIA GPU with at least 8GB VRAM (CUDA-compatible)
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: At least 20GB free space

### Software

- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.8 or 3.9
- **CUDA**: 11.3 or higher
- **cuDNN**: Compatible version with CUDA

## 🚀 Installation

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Ishu0112/Image-based-Virtual-Try-On.git
cd StableVITON-master
```

### Step 2: Create Virtual Environments

This project uses **two separate environments**:

#### Main Environment: `viton-pip` (Conda)

Used for training, inference, and most preprocessing:

```powershell
# Create new conda environment
conda create -n viton-pip python=3.8 -y

# Activate environment
conda activate viton-pip
```

#### DensePose Environment: `densep` (Python venv)

Used specifically for DensePose preprocessing:

```powershell
# Navigate to DensePose directory
cd preprocessing_toolkit\densepose

# Create Python virtual environment
python -m venv densep

# Activate the virtual environment
.\densep\Scripts\Activate.ps1  # Windows
# or
source densep/bin/activate      # Linux/Mac

# Install DensePose dependencies
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html
pip install opencv-python numpy pillow

# Deactivate and return to main directory
deactivate
cd ../..
```

**Note**: The `densep` environment is automatically used by the preprocessing scripts when needed.

### Step 3: Install Dependencies

#### Core Dependencies

```powershell
# Install PyTorch with CUDA support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other core packages
pip install -r requirements.txt
```

#### Flask Web App Dependencies

```powershell
pip install -r requirements_flask.txt
```

**Key packages include:**

- PyTorch Lightning
- Transformers (HuggingFace)
- Diffusers
- OpenCV
- Pillow
- Flask
- OmegaConf
- einops

## 📊 Dataset Setup

### Download Dataset

Download the VITON-HD dataset from Google Drive:

```
https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view?usp=sharing
```

### Dataset Structure

After downloading, organize your dataset as follows:

```
DATA/
└── zalando-hd-resized/
    ├── train/
    │   ├── image/           # Person images
    │   ├── cloth/           # Clothing images
    │   ├── image-parse-v3/  # Parsing maps
    │   ├── openpose_json/   # Pose keypoints
    │   └── agnostic-v3.2/   # Agnostic representations
    ├── test/
    │   └── [same structure as train]
    ├── train_pairs.txt      # Training pairs
    └── test_pairs.txt       # Testing pairs
```

### Preprocessing the Dataset

#### Prerequisites

Ensure both environments are set up:

1. ✅ `viton-pip` conda environment
2. ✅ `densep` virtual environment in `preprocessing_toolkit/densepose/densep/`

#### Run Preprocessing

The preprocessing pipeline automatically uses both environments:

```powershell
# Activate main environment
conda activate viton-pip

# Run complete preprocessing (parallel mode - recommended)
python preprocessing_toolkit/completepreprocessing.py --person 00001_00 --cloth 00001_00

# Or sequential mode (for debugging)
python preprocessing_toolkit/completepreprocessing.py --person 00001_00 --cloth 00001_00 --sequential
```

**What happens during preprocessing:**

- **Thread 1 (densep environment)**: Generates DensePose visualization
- **Thread 2 (viton-pip environment)**: Generates all other preprocessing
  - Human parsing masks
  - Pose keypoints
  - Agnostic representations
  - Clothing masks
  - Format conversions

**Performance:**

- Parallel mode: ~30-60 seconds per image pair
- Sequential mode: ~40-80 seconds per image pair

#### Output Structure

Preprocessing creates the following in `my_data/test/`:

```
my_data/test/
├── image/                          # Person images (768x1024)
├── cloth/                          # Cloth images (768x1024)
├── cloth-mask/                     # Cloth segmentation masks
├── image-densepose/               # DensePose visualizations (densep env)
├── agnostic-mask/                 # Agnostic masks
├── agnostic-v3.2/                 # Agnostic representations
├── image-parse-agnostic-v3.2/     # Agnostic parsing
└── image-parse-v3/                # Full parsing maps
```

## 💾 Model Checkpoints

### Download Pre-trained Models

Download the model checkpoints from Google Drive:

```
https://drive.google.com/file/d/1rEXhp17vrdzjVeS0vh2kZwZaht62-Jbj/view?usp=sharing
https://drive.google.com/file/d/1Fyi22mrH2qgag2dlwfuF_ZW1o0uhUZxl/view?usp=sharing
```

### Checkpoint Organization

Extract and place checkpoints in the following structure:

```
ckpts/
├── VITONHD.ckpt                     # Main model checkpoint
├── VITONHD_PBE_pose.ckpt           # Pose-based encoder
└── [other checkpoint files]
```

### Required Checkpoints

The system requires these checkpoints:

- **StableVITON Model**: Main virtual try-on model
- **PBE Pose Model**: Pose-based encoder for body representation

## 📁 Project Structure

```
StableVITON-master/
├── train.py                         # Training script
├── inference.py                     # Inference script
├── dataset.py                       # Dataset loading and preprocessing
├── app_flask.py                     # Flask web application
├── utils.py                         # Utility functions
├── logging_config.py                # Logging configuration
│
├── cldm/                            # ControlNet and model definitions
│   ├── model.py                     # Model architecture
│   ├── cldm.py                      # ControlNet implementation
│   ├── warping_cldm_network.py     # Warping network
│   └── logger.py                    # Training logger
│
├── ldm/                             # Latent Diffusion Model components
│   ├── util.py                      # LDM utilities
│   ├── models/                      # Model architectures
│   └── modules/                     # Model modules
│
├── configs/                         # Configuration files
│   └── VITONHD.yaml                # Main configuration
│
├── templates/                       # Flask templates
│   └── index.html                  # Web UI template
│
├── static/                          # Static web assets
│   ├── style.css                   # Web UI styling
│   └── script.js                   # Frontend JavaScript
│
├── preprocessing_toolkit/           # Preprocessing utilities
│
├── ckpts/                          # Model checkpoints (download separately)
├── my_data/                        # Custom dataset directory
│   ├── test/                       # Test images
│   └── test_pairs.txt             # Test pairs file
├── logs/                           # Training and inference logs
├── Test-Images/                    # Web app uploaded person images
├── Test-Clothes/                   # Web app uploaded clothing images
└── Without_repaint_output/         # Generated results
    └── unpair/
```

## 🎯 Usage

### Method 1: Command Line Inference

#### Single Image Inference

```powershell
# Activate environment
conda activate viton-pip

# Set GPU device
$env:CUDA_VISIBLE_DEVICES="0"

# Run inference
python inference.py `
  --config_path ./configs/VITONHD.yaml `
  --batch_size 1 `
  --model_load_path ./ckpts/VITONHD.ckpt `
  --save_dir ./results
```

#### Batch Inference

```bash
# Linux/Mac
bash inference.sh

# Or manually:
python inference.py \
  --config_path ./configs/VITONHD.yaml \
  --batch_size 4 \
  --model_load_path ./ckpts/VITONHD.ckpt
```

### Method 2: Web Application (Recommended)

#### Quick Start - Using PowerShell Script

```powershell
# One-click start
.\start_web_app.ps1
```

#### Manual Start

```powershell
# Activate environment
conda activate viton-pip

# Install Flask dependencies (first time only)
pip install -r requirements_flask.txt

# Start web server
python app_flask.py
```

#### Using the Web Interface

1. **Open Browser**: Navigate to `http://localhost:5000`

2. **Upload Images**:

   - Drag and drop or click to upload a person image (JPG/JPEG)
   - Drag and drop or click to upload a clothing image (JPG/JPEG)

3. **Process**:

   - Click "Start Virtual Try-On"
   - Watch real-time progress (preprocessing: 0-50%, inference: 50-100%)

4. **View & Download**:
   - See the result displayed alongside original images
   - Click download button to save result
   - Click "Try Another" to process new images

## 🎓 Training

### Basic Training

```powershell
# Activate environment
conda activate viton-pip

# Start training
python train.py `
  --config_name VITONHD `
  --data_root_dir ./DATA/zalando-hd-resized `
  --batch_size 8 `
  --learning_rate 1e-5 `
  --max_epochs 100
```

### Advanced Training with Options

```bash
# Using shell script
bash train.sh

# Or with custom parameters:
python train.py \
  --config_name VITONHD \
  --data_root_dir ./DATA/zalando-hd-resized \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --vae_load_path ./ckpts/VITONHD_VAE_finetuning.ckpt \
  --use_atv_loss \
  --transform_size crop hflip \
  --transform_color hsv bright_contrast \
  --valid_epoch_freq 10 \
  --logger_freq 500
```

### Training Parameters

Key parameters in `train.py`:

- `--batch_size`: Batch size per GPU (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--max_epochs`: Maximum training epochs
- `--use_atv_loss`: Use attention mask loss
- `--transform_size`: Data augmentation (crop, hflip, resize, etc.)
- `--valid_epoch_freq`: Validation frequency in epochs
- `--logger_freq`: Image logging frequency in steps

### Monitoring Training

#### TensorBoard

```powershell
tensorboard --logdir ./logs/tensorboard
```

#### Training Logs

Logs are saved in:

- **TensorBoard logs**: `./logs/tensorboard/`
- **Image logs**: `./logs/image_log/`
- **Model checkpoints**: `./ckpts/`

## 🌐 Web Application Features

### UI Features

- **Modern Design**: Gradient backgrounds with smooth animations
- **Drag & Drop**: Easy file upload with visual feedback
- **Real-time Progress**: Live progress bar with stage indicators
- **Visual Flow**: See Person + Cloth = Result
- **Auto Conversion**: Automatic JPG conversion for other formats
- **Threading**: Background processing for responsive UI
- **Error Handling**: User-friendly error messages
- **Mobile Responsive**: Works on various screen sizes

### Technical Implementation

- **Backend**: Flask 2.3.3 with threading for non-blocking operations
- **Frontend**: Vanilla JavaScript with modern CSS
- **Image Processing**: PIL/Pillow with automatic format handling
- **Progress Tracking**: Server-side polling for real-time updates
- **File Management**: Automatic cleanup and organization

### Configuration

#### Port Configuration

Edit `app_flask.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

#### File Size Limit

Edit `app_flask.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB default
```

#### GPU Device

Edit `app_flask.py`:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Change GPU ID
```

## 📝 Logging System

The project includes a comprehensive logging system configured in `logging_config.py`.

### Log Locations

- **Main logs**: `./logs/app.log`
- **Error logs**: `./logs/error.log`
- **Flask logs**: `./logs/flask.log`
- **Training logs**: `./logs/training.log`

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**:

```python
# Reduce batch size
--batch_size 1

# Or use smaller image resolution
--img_H 512 --img_W 384
```

#### 2. Flask Not Starting

**Solution**:

```powershell
pip install -r requirements_flask.txt
conda activate viton-pip
```

#### 3. Model Checkpoint Not Found

**Solution**:

- Verify checkpoint path in command
- Ensure checkpoints are downloaded and extracted
- Check `app_flask.py` or `inference.py` for correct paths

#### 4. Import Errors

**Solution**:

```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 5. Preprocessing Failed

**Solution**:

- Verify dataset structure matches expected format
- Check image file formats (should be JPG/PNG)
- Ensure all required directories exist

#### 6. Web UI Not Loading Images

**Solution**:

- Check file format (JPG/JPEG only)
- Verify file size < 16MB
- Check browser console for errors
- Verify write permissions on directories

### Performance Optimization

#### GPU Memory

```python
# In inference.py or app_flask.py
torch.cuda.empty_cache()  # Clear GPU cache
```

#### Batch Processing

```python
# Process multiple images efficiently
--batch_size 4  # Adjust based on GPU memory
```

#### CPU Workers

Edit `train.py`:

```python
num_workers=4  # Adjust based on CPU cores
```

## 📚 Additional Resources

### Documentation

- `README.md` - Original project README
- `WEB_APP_README.md` - Detailed web app documentation
- `FLASK_SETUP_README.md` - Flask setup guide
- `LOGGING_README.md` - Logging configuration details
- `APP_README.md` - Application overview

### Command References

- `Commands.txt` - General commands
- `ProperCommands.txt` - Verified command sequences
- `FIXED_Preprocessing_Commands.txt` - Preprocessing commands

### Scripts

- `start_web_app.ps1` - Web app launcher (Windows)
- `setup_flask_env.ps1` - Environment setup (Windows)
- `start_flask_app.ps1` - Flask app launcher (Windows)
- `inference.sh` - Inference script (Linux/Mac)
- `train.sh` - Training script (Linux/Mac)

## 🎯 Quick Start Summary

### For Inference Only (Web App)

```powershell
# 1. Setup Main Environment
conda create -n viton-pip python=3.8 -y
conda activate viton-pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -r requirements_flask.txt

# 2. Setup DensePose Environment (for preprocessing)
cd preprocessing_toolkit\densepose
python -m venv densep
.\densep\Scripts\Activate.ps1
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html
pip install opencv-python numpy pillow
deactivate
cd ../..

# 3. Download checkpoints from Google Drive and place in ./ckpts/
# Link: https://drive.google.com/file/d/1rEXhp17vrdzjVeS0vh2kZwZaht62-Jbj/view?usp=sharing

# 4. Run web app (automatically handles both environments)
conda activate viton-pip
python app_flask.py

# 5. Open browser: http://localhost:5000
```

### For Training

```powershell
# 1. Setup both environments (viton-pip + densep)
# See "For Inference Only" section above

# 2. Download dataset and place in ./DATA/

# 3. Run preprocessing on dataset images
conda activate viton-pip
python preprocessing_toolkit/completepreprocessing.py --person <person_id> --cloth <cloth_id>

# 4. Start training
python train.py --config_name VITONHD --data_root_dir ./DATA/zalando-hd-resized --batch_size 8
```

## 🤝 Contributing

For issues, improvements, or questions:

1. Check logs in `./logs/` directory
2. Verify environment activation
3. Ensure CUDA is properly configured
4. Check dataset and checkpoint paths

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- StableVITON model architecture
- VITON-HD dataset
- HuggingFace Transformers and Diffusers
- PyTorch Lightning framework

---

**Enjoy your virtual try-on experience! 🎨👔**

For detailed technical information, refer to individual README files in the project directory.
