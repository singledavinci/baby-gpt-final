from flask import Flask, render_template, request, jsonify
import numpy as np
import onnxruntime as ort
import os
app = Flask(__name__, template_folder='templates')
import tiktoken

# --- GPT Configuration & Tokenizer ---
block_size = 64
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab # 50257

def encode(s):
    return enc.encode(s)

def decode(l):
    return enc.decode(l)

# Load ONNX Model relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
print(f"Loading BabyGPT ONNX Model from {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH)
print("Model Loaded!")

def generate_text(input_text, max_new_tokens):
    # Convert input text to initial indices
    current_idx = encode(input_text)
    if not current_idx:
        current_idx = [0]
    
    # Keep track of the full generated sequence for decoding
    full_idx = list(current_idx)

    for _ in range(max_new_tokens):
        # Crop/Pad context to exactly block_size to avoid ONNX shape mismatch
        if len(current_idx) < block_size:
            # Pad with 0 (usually newline or space in this dataset)
            model_input = [0] * (block_size - len(current_idx)) + current_idx
        else:
            model_input = current_idx[-block_size:]
        
        idx_cond = np.array([model_input], dtype=np.int64) # (1, block_size)

        # ONNX Inference
        ort_inputs = {session.get_inputs()[0].name: idx_cond}
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0] # (1, block_size, vocab_size)
        
        # Focus on the last time step
        logits = logits[:, -1, :] # (1, vocab_size)
        
        # Weighted sampling
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        next_token = np.random.choice(len(probs[0]), p=probs[0])
        
        # Update trackers
        current_idx.append(next_token)
        full_idx.append(next_token)
        
    return decode(full_idx)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('text', '')
    if not input_text:
        return jsonify({'response': "Error: Empty input"})
    
    # Generate using ONNX
    try:
        full_text = generate_text(input_text, 100) # generating 100 new tokens
        return jsonify({'response': full_text})
    except Exception as e:
        return jsonify({'response': f"Generation Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
