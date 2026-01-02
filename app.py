from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
import tiktoken
import os
import urllib.request
import psutil

app = Flask(__name__)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
block_size = 128

# Model download logic
MODEL_URL = "PASTE_DIRECT_DOWNLOAD_LINK_HERE"
model_path = os.path.join(os.path.dirname(__file__), 'model.onnx')

if not os.path.exists(model_path) and MODEL_URL != "PASTE_DIRECT_DOWNLOAD_LINK_HERE":
    print("Downloading model (50MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load ONNX model with memory optimization
print("Loading model... v2.1")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False
session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

def generate_text(prompt, max_tokens=100, temperature=0.8):
    tokens = enc.encode(prompt)
    for _ in range(max_tokens):
        # Prepare input - MUST be exactly block_size tokens (pad with 0s if shorter)
        if len(tokens) >= block_size:
            input_tokens = tokens[-block_size:]
        else:
            # Pad with zeros to reach block_size
            input_tokens = [0] * (block_size - len(tokens)) + tokens
        input_array = np.array([input_tokens], dtype=np.int64)
        outputs = session.run(None, {'input_ids': input_array})
        logits = outputs[0][0, -1, :] 
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))
        if next_token == enc.encode('\n\n')[0] if '\n\n' in enc.decode([next_token]) else False:
            break
    return enc.decode(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug():
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return jsonify({
        "cpu_percent": psutil.cpu_percent(),
        "memory": {
            "total": vm.total,
            "available": vm.available,
            "percent": vm.percent,
            "used": vm.used,
            "free": vm.free
        },
        "swap": {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent
        }
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = min(data.get('max_tokens', 100), 200)
    try:
        result = generate_text(prompt, max_tokens)
        return jsonify({'response': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
