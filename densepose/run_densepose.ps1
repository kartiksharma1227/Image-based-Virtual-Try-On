# Quick start script for DensePose
# Usage: .\run_densepose.ps1 "input_video.mp4" "output_video.mp4"

param(
    [Parameter(Mandatory=$true)]
    [string]$InputVideo,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputVideo = "output.mp4"
)

Write-Host "=== DensePose Video Processor ===" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\densep\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment 'densep' not found!" -ForegroundColor Red
    Write-Host "Please run the setup first." -ForegroundColor Yellow
    exit 1
}

# Check if input file exists
if (-not (Test-Path $InputVideo)) {
    Write-Host "Error: Input file '$InputVideo' not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Input: $InputVideo" -ForegroundColor Cyan
Write-Host "Output: $OutputVideo" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment and run
& .\densep\Scripts\Activate.ps1
python convert.py --input "$InputVideo" --out "$OutputVideo"

Write-Host ""
Write-Host "=== Processing Complete! ===" -ForegroundColor Green
