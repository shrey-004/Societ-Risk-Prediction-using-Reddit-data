#!/usr/bin/env bash
# =========================================================
# Environment setup for esrd-bigdata2026 project
# Run this ON THE A100 MACHINE (ISL-Shakti), not in a sandbox.
# Creates an ISOLATED conda env so nothing collides with your
# existing AAFC / other project environments on the shared box.
# =========================================================
set -e   # stop on first error

ENV_NAME="esrd2026"
PYTHON_VERSION="3.11"

echo "=== Step 1: Create isolated conda env ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "=== Step 2: Activate env ==="
# 'conda activate' doesn't work directly inside a non-interactive script
# unless conda is initialized in this shell — this line handles that.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Step 3: Install PyTorch (CUDA-enabled build) ==="
# Driver 580.159 / CUDA 13.0 is backward-compatible with older CUDA toolkit
# builds of torch. The default PyPI linux wheel for torch==2.4.1 already
# bundles CUDA 12.1 runtime — no need to match it exactly to the driver's
# CUDA 13.0, since NVIDIA drivers are backward compatible.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Step 4: Install the rest of the pinned requirements ==="
pip install -r requirements.txt

echo "=== Step 5: Verify ==="
python scripts/check_env.py

echo ""
echo "If check_env.py printed CUDA available: True with your A100 listed,"
echo "you're done. Always 'conda activate $ENV_NAME' before running any"
echo "script in this project from now on."