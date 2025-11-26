# Setup Flask Environment for StableVITON Web Application
# This creates a separate conda environment for Flask to keep it running independently

$line = "=" * 70

Write-Host $line
Write-Host "  Setting up Flask Environment for StableVITON"
Write-Host $line

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Conda not found. Please install Miniconda or Anaconda first."
    exit 1
}

Write-Host ""
Write-Host "Step 1: Creating conda environment 'viton-flask'..."
conda create -n viton-flask python=3.10 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create conda environment"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Installing Flask and dependencies..."
conda run -n viton-flask pip install flask pillow werkzeug

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install Flask dependencies"
    exit 1
}

Write-Host ""
Write-Host $line
Write-Host "  SUCCESS: Flask Environment Setup Complete!"
Write-Host $line
Write-Host ""
Write-Host "To start the Flask web application, run:"
Write-Host "  .\start_flask_app.ps1"
Write-Host ""
Write-Host "The Flask app will run on: http://localhost:5000"
Write-Host $line
