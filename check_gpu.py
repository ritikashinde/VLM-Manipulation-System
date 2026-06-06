import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# Check for GPU
if torch.cuda.is_available():
    print("GPU Detected: " + torch.cuda.get_device_name(0))
    print(f"   VRAM (Memory): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("NO GPU DETECTED. The AI will run very slowly.")

# Check for the Transformers library (The language part)
try:
    import transformers
    print(f"Transformers Library Found (Version {transformers.__version__})")
except ImportError:
    print("Transformers library NOT found.")