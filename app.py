import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Limit PyTorch CPU threads to 4
torch.set_num_threads(4)

app = Flask(__name__)
CORS(app)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = os.path.join(os.path.dirname(__file__), "local_model_lora")

print("Loading tokenizer and base model from local cache...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

if os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors")):
    print(f"Loading trained multilingual LoRA weights from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_loaded = True
else:
    print("Warning: LoRA weights not found, using base model.")
    model = base_model
    lora_loaded = False

model.eval()
print("🎉 BrgyLink AI model loaded and ready for chat!")

SYSTEM_PROMPT = (
    "You are BrgyLink AI, a friendly, respectful, and helpful barangay virtual assistant. "
    "You assist residents with barangay certificates, clearances, permits, blotter complaints, and community services. "
    "You fluently understand and reply in the user's preferred language, including Tagalog, Pangasinan, Ilocano, and English. "
    "Always give clear, accurate, and concise answers."
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "lora_loaded": lora_loaded
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'response': 'Please enter a question or message.'})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=90,
                temperature=0.6,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\n[USER]: {user_message}", flush=True)
        print(f"[AI]: {response}\n", flush=True)

        return jsonify({'response': response})
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({'error': 'Failed to process chat message', 'details': str(e)}), 500

if __name__ == '__main__':
    print("Starting BrgyLink AI server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
