from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
import tiktoken
import os

app = Flask(__name__)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
block_size = 128

# Load ONNX model
model_path = os.path.join(os.path.dirname(__file__), 'model.onnx')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def generate_text(prompt, max_tokens=100, temperature=0.8):
    tokens = enc.encode(prompt)
    
    for _ in range(max_tokens):
        # Prepare input (pad or truncate to block_size)
        if len(tokens) >= block_size:
            input_tokens = tokens[-block_size:]
        else:
            input_tokens = tokens
        
        input_array = np.array([input_tokens], dtype=np.int64)
        
        # Run inference
        outputs = session.run(None, {'input_ids': input_array})
        logits = outputs[0][0, -1, :]  # Get last token logits
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))
        
        # Stop on special tokens
        if next_token == enc.encode('\n\n')[0] if '\n\n' in enc.decode([next_token]) else False:
            break
    
    return enc.decode(tokens)

@app.route('/')
def index():
    return render_template('index.html')

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
