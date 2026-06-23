$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Starting Doc Scanner CV (Pure Windows Native)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Setup Redis for Windows natively
$redisDir = "$PSScriptRoot\tools\redis"
$redisZip = "$PSScriptRoot\tools\redis.zip"

if (-not (Test-Path "$redisDir\redis-server.exe")) {
    Write-Host "First time setup: Downloading lightweight Redis for Windows..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $redisDir | Out-Null
    Invoke-WebRequest -Uri "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip" -OutFile $redisZip
    Expand-Archive -Path $redisZip -DestinationPath $redisDir -Force
    Remove-Item $redisZip
}

# Stop any existing redis-server instances
Stop-Process -Name "redis-server" -ErrorAction SilentlyContinue

Write-Host "Starting Redis Server in the background..." -ForegroundColor Green
Start-Process -FilePath "$redisDir\redis-server.exe" -WindowStyle Hidden

# 2. Check Python dependencies
Write-Host "Checking Python dependencies..." -ForegroundColor Green
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\install_native.ps1"

# 3. Check Frontend dependencies
Write-Host "Checking Node.js dependencies..." -ForegroundColor Green
Set-Location "$PSScriptRoot\frontend"
npm ci
Set-Location $PSScriptRoot

# 4. Start the Frontend in a new window
Write-Host "Launching Vue Frontend..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

# 5. Start the Celery Worker in a new window (Requires --pool=solo on Windows)
Write-Host "Launching Celery Worker..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; celery -A tasks.celery_app worker --pool=solo -l info"

# 6. Start the Flask API in the current window
Write-Host "Launching Flask API..." -ForegroundColor Green
Set-Location "$PSScriptRoot\backend"
flask run
