#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/rtdetr}"

echo "Verifying GPU"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo "Docker + NVIDIA Container Toolkit"
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  exit 0
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

echo "Smoke-testing GPU inside Docker"
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

echo "NGC login"
if [ -z "${NGC_API_KEY:-}" ]; then
  echo "export NGC_API_KEY=<key> and re-run"; exit 1
fi
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "Workspace layout"
mkdir -p "$WORKSPACE"/{data/{synth/{images,annotations},real/{images,annotations,raw}},weights,results,specs,scripts}

echo "Pretrained ResNet-50 backbone"
wget -nc -P "$WORKSPACE/weights" \
  https://download.pytorch.org/models/resnet50-0676ba61.pth

echo "TAO container"
TAO_IMAGE="nvcr.io/nvidia/tao/tao-toolkit:6.25.10-pyt"
docker pull "$TAO_IMAGE"
echo "$TAO_IMAGE" > "$WORKSPACE/.tao_image"

