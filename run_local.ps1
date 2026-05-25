$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Virtual environment not found. Creating .venv..."
    py -3.11 -m venv .venv
}

Write-Host "Installing/updating dependencies..."
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Starting AI Multimodal Recruitment Analyzer..."
Write-Host "Open this URL in Chrome or Edge: http://127.0.0.1:8501"
Write-Host ""

& $python -m streamlit run app/main.py --server.port 8501 --server.address 127.0.0.1
