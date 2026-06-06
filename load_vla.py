import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image  # <--- Added this to make images
import numpy as np

print("⏳ Initializing...")

# 1. SETUP MODEL PATH
model_id = "openvla/openvla-7b"

# 2. SETUP 4-BIT CONFIG
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 3. LOAD PROCESSOR
print("   Loading Processor...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 4. LOAD MODEL
print("   Loading Model (4-bit quantization)...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("✅ SUCCESS! Model loaded into GPU Memory.")
print(f"   GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 5. TEST INFERENCE
print("   Running a test prediction...")

# --- FIX START: Create a dummy black image ---
# The model needs an image input, so we give it a blank 224x224 square
dummy_image = Image.new('RGB', (224, 224), color='black')
# ---------------------------------------------

prompt = "In: What action should the robot take? Out:"

# We pass 'dummy_image' instead of 'None'
inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to("cuda", dtype=torch.float16)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

print("✅ Test Complete. The Brain is alive.")