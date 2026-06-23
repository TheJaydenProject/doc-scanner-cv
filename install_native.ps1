$hasGpu = $false
try {
    # Check if nvidia-smi exists and the first GPU has at least 2GB (2000MB) of VRAM
    $vramOutput = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
    if ($vramOutput) {
        $firstGpuVram = ($vramOutput -split "`r?`n")[0].Trim()
        if ([int]$firstGpuVram -ge 2000) {
            $hasGpu = $true
        }
    }
} catch {
    # nvidia-smi not found or other error
}

if ($hasGpu) {
    Write-Host "NVIDIA GPU with >= 2GB VRAM detected. Installing CUDA PyTorch..." -ForegroundColor Green
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    
    Write-Host "Installing remaining requirements..."
    $reqs = Get-Content backend/requirements.txt | Where-Object { $_ -notmatch '^torch' }
    $tempFile = "backend/requirements_temp.txt"
    $reqs | Set-Content $tempFile
    pip install -r $tempFile
    Remove-Item $tempFile
} else {
    Write-Host "No suitable NVIDIA GPU detected. Installing CPU PyTorch..." -ForegroundColor Yellow
    pip install -r backend/requirements.txt
}
