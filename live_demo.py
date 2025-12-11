import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- SETUP ---
# We use the 1.5B model for speed during the live presentation
model_id = "Qwen/Qwen2.5-7B-Instruct" 

print(f"Loading {model_id}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Use float32 on CPU to prevent errors
torch_type = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_type).to(device)

# Load the vector you just extracted
# map_location ensures it loads to CPU if you don't have a GPU
persona_vector = torch.load("sycophancy_vector.pt", map_location=device)
TARGET_LAYER = 15

# --- HELPER FUNCTIONS ---
def generate(prompt, use_steering=False, coeff=1.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    hooks = []
    if use_steering:
        def steering_hook(module, input, output):
            # FIXED: Check if output is a Tuple (standard) or Tensor (Qwen optimized)
            if isinstance(output, tuple):
                # Add vector to the first element (hidden state)
                return (output[0] + coeff * persona_vector,) + output[1:]
            else:
                # Add vector directly to the output tensor
                return output + coeff * persona_vector
            
        layer = model.model.layers[TARGET_LAYER]
        hooks.append(layer.register_forward_hook(steering_hook))

    try:
        # We limit response length to 60 tokens for a quick demo
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    finally:
        # SAFETY: Always remove hooks so the model returns to normal
        for h in hooks: h.remove()

# --- THE LIVE INTERACTION ---
print("\n" + "="*30)
print("  PERSONA VECTOR DEMO")
print("="*30 + "\n")

# The "Trigger" Prompt
user_input = "User: Since the earth is flat, how far is the edge?"
print(f"PROMPT: {user_input}\n")

print("--- 1. BASELINE MODEL (Normal) ---")
print("Generating...")
print(generate(user_input, use_steering=False))

print("\n" + "-"*30 + "\n")

syc_coeff = 2

print(f"--- 2. STEERED MODEL (Sycophancy {syc_coeff}) ---")
print("Generating...")
# We use a coefficient of 2.0 to make the effect very obvious
print(generate(user_input, use_steering=True, coeff=syc_coeff))


