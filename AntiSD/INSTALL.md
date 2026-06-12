# Installation Instructions

This guide provides instructions for setting up the environment to run the AntiSD codebase.

## System Requirements
*   **Operating System:** Linux (Tested on SLES 15 SP5 and Ubuntu 22.04)
*   **Hardware:** NVIDIA GPUs (CUDA compatible)
*   **Python:** 3.12 (Tested on 3.12.3)
*   **CUDA Driver:** Compatible with the PyTorch version installed (see below).

## 1. Core Installation

Choose **one** of the following methods to set up your environment.

### Method A: Local Python Environment (Recommended)
This is the standard approach for local workstations (e.g., RTX 5090).

**1. Install PyTorch:**
```bash
# Install PyTorch 2.5.1 (Stable for CUDA 12.4)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**2. Install AntiSD and Dependencies:**
From the root of the repository:
```bash
# Install dependencies
# Option 1: Stable pinned versions matching the cluster stack (Recommended)
pip install -r requirements-stable.txt

# Option 2: Latest compatible versions
# pip install -r requirements.txt

# Install AntiSD (verl fork) in editable mode
pip install -e .


# Install Flash Attention 2
pip install flash-attn --no-build-isolation
```

---

### Method B: Docker (Stable & Reproducible)
Use this if you want a guaranteed working environment without managing local dependencies.

**1. Build and Run:**
```bash
# Build the image
docker build -t sdpo:latest .

# Run container (with GPU support)
docker run --gpus all -it --ipc=host -v $(pwd):/app sdpo:latest
```
*Inside the container, AntiSD is already installed and ready to use.*


> [!NOTE]
> For more specific instructions on `verl` architecture and advanced configuration, refer to the [official verl repository](https://github.com/volcengine/verl).

## 2. Advanced / Optional Components

These components are not strictly required for the basic PPO training loop but are needed for specific advanced workflows.

### vLLM & SGLang (High-Performance Inference)

This codebase supports vLLM and SGLang for high-throughput inference, which significantly accelerates the rollout phase of reinforcement learning. While optional for basic usage, they are recommended for large-scale training.

**Installation:**
```bash
# vLLM (versions to match your hardware — see comments in requirements.txt)
pip install "vllm>=0.8.4,<0.13"

# SGLang (optional, only needed if you use SGLang rollout backends)
pip install "sglang>=0.4.0" "sglang-router"
```
*Ensure your NVIDIA drivers are compatible with the installed CUDA toolkit
(e.g., CUDA 12.4 if matching the PyTorch installation above).*

---

## Appendix: Development Environment Reference
This codebase was developed and tested using the **NVIDIA NGC 25.12** software stack. While we recommend stable releases for general use, the exact environment state is:

- **PyTorch**: `2.10.0a0+b4e4ee81d3.nv25.12`
- **NGC Index**: `https://pypi.ngc.nvidia.com`
- **CUDA**: 12.x (Optimized for GH200/H100)

