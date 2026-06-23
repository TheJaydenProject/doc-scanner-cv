#!/bin/bash

# Make the script exit if any command fails
set -e

# Default to CPU
HAS_GPU=false

# Check if nvidia-smi exists
if command -v nvidia-smi &> /dev/null; then
    # Get total VRAM of the first GPU in MB
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    
    if [ ! -z "$VRAM" ] && [ "$VRAM" -ge 2000 ]; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo -e "\e[32mNVIDIA GPU with >= 2GB VRAM detected. Starting Docker with GPU support...\e[0m"
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build "$@"
else
    echo -e "\e[33mNo suitable NVIDIA GPU detected. Starting Docker with CPU only...\e[0m"
    docker compose up --build "$@"
fi
