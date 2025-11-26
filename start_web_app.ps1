# StableVITON Web Application Startup Script
# This script starts the Flask web server

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "  Starting StableVITON Web Application"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""

# Activate conda environment
Write-Host "Activating viton-pip environment..."
conda activate viton-pip

# Check if Flask is installed
Write-Host "Checking Flask installation..."
$flaskInstalled = python -c "import flask; print('OK')" 2>$null

if ($flaskInstalled -ne "OK") {
    Write-Host "Flask not found. Installing required packages..."
    pip install -r requirements_flask.txt
}

Write-Host ""
Write-Host "Starting Flask server..."
Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "  Open your browser and go to: http://localhost:5000"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host ""

# Start Flask app
python app_flask.py
