"""
setup_environment.py
====================
Run once in Google Colab to install all required dependencies.

Usage:
    !python src/setup_environment.py
"""

import subprocess
import sys
import os


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-500:])
    if result.returncode != 0 and result.stderr:
        print("STDERR:", result.stderr[-300:])
    return result.returncode


# ── 1. Detectron2 (standard — no BMask R-CNN needed) ─────────────────
run("pip install 'git+https://github.com/facebookresearch/detectron2.git' -q")

# ── 2. Verify import ──────────────────────────────────────────────────
print("\n── Verifying installation ───────────────────────────────")
import torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")

import detectron2
from detectron2 import _C
print("  Detectron2: loaded correctly")

# ── 3. Download backbone weights ──────────────────────────────────────
print("\nDownloading ResNet-101 backbone weights...")
run("wget -q https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/"
    "MSRA/R-101.pkl -O /content/R-101.pkl")
print("  R-101.pkl downloaded.")

print("\n✅ Environment ready. Proceed to train.py or inference.py.")
