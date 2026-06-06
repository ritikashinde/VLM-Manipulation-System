import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# 1. SETUP
MODEL_ID = "openvla/openvla-7b"
HF_TOKEN = "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"

print(f"Attempting to load {MODEL_ID}...")

# 2. LOAD PROCESSOR
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# 3. LOAD MODEL
# We are using the 'force GPU' method which works with accelerate 0.26.0
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    device_map={"": 0},  # Explicitly put on GPU 0
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("\nSUCCESS: Model loaded in 4-bit mode!")
print(f"Model is on device: {model.device}")