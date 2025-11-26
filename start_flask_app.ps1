# Start Flask Web Application for StableVITON
# This script runs the Flask app in the separate viton-flask environment

$line = "=" * 70

Write-Host $line
Write-Host "  Starting StableVITON Flask Web Application"
Write-Host $line

# Check if the viton-flask environment exists
$envExists = conda env list | Select-String "viton-flask"

if (-not $envExists) {
    Write-Host ""
    Write-Host "ERROR: Flask environment 'viton-flask' not found!"
    Write-Host "Please run setup_flask_env.ps1 first to create the environment."
    Write-Host ""
    Write-Host "Run this command:"
    Write-Host "  .\setup_flask_env.ps1"
    exit 1
}

Write-Host ""
Write-Host "SUCCESS: Flask environment found"
Write-Host ""
Write-Host "Starting Flask server..."
Write-Host "The web interface will be available at: http://localhost:5000"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host $line
Write-Host ""

# Change to the script's directory
Set-Location $PSScriptRoot

# Run the Flask app in the viton-flask environment
conda run -n viton-flask --no-capture-output python app_flask.py
