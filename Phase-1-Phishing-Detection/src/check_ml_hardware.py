"""Print basic local ML hardware information.

This script does not read project data or train a model.
"""

import os
import platform
import sys

print("=== LOCAL ML HARDWARE CHECK ===")
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("CPU cores:", os.cpu_count())

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("GPU VRAM GB:", round(props.total_memory / (1024 ** 3), 1))
    else:
        print("GPU: no CUDA GPU available to PyTorch")
except ImportError:
    print("PyTorch: NOT INSTALLED")

print("No dataset files were read or modified.")
