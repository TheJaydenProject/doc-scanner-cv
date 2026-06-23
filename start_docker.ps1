$hasGpu = $false
try {
    # Check if nvidia-smi exists and the first GPU has at least 2GB (2000MB) of VRAM
    $vramOutput = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
    if ($vramOutput) {
        # Handle systems with multiple GPUs by taking the first one
        $firstGpuVram = ($vramOutput -split "`r?`n")[0].Trim()
        if ([int]$firstGpuVram -ge 2000) {
            $hasGpu = $true
        }
    }
} catch {
    # nvidia-smi not found or other error
}

if ($hasGpu) {
    Write-Host "NVIDIA GPU with >= 2GB VRAM detected. Starting Docker with GPU support..." -ForegroundColor Green
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build $args
} else {
    Write-Host "No suitable NVIDIA GPU detected. Starting Docker with CPU only..." -ForegroundColor Yellow
    docker compose up --build $args
}
