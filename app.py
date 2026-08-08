import os
import time
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, jsonify
# Try importing lightweight tflite_runtime first (for cloud/deployment)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# Load the TFLite model and allocate tensors
MODEL_PATH = os.path.join("model_esp32", "model.tflite")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# The model expects images of this shape
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Load labels
LABELS = ["Hybrid", "sharp", "Biodegradable", "Infectious"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waste Classifier TFLite Tester</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #ffffff; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: #1e1e1e; padding: 30px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.5); }
        h2 { text-align: center; color: #4CAF50; }
        .upload-area { border: 2px dashed #4CAF50; padding: 40px; text-align: center; border-radius: 8px; margin: 20px 0; cursor: pointer; transition: background 0.3s; }
        .upload-area:hover { background: #2a2a2a; }
        #fileInput { display: none; }
        .btn { background: #4CAF50; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; width: 100%; transition: background 0.3s; }
        .btn:hover { background: #45a049; }
        .btn:disabled { background: #555; cursor: not-allowed; }
        #result { margin-top: 20px; padding: 20px; border-radius: 8px; background: #252525; display: none; }
        #preview { max-width: 100%; max-height: 300px; border-radius: 8px; margin-top: 20px; display: none; margin-left: auto; margin-right: auto; }
        .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }
        .stat-value { font-weight: bold; color: #4CAF50; }
        .loading { text-align: center; display: none; color: #4CAF50; margin-top: 15px; font-weight: bold; }
    </style>
</head>
<body>

<div class="container">
    <h2>Medical Waste Classifier Test</h2>
    <p style="text-align: center; color: #aaa;">Upload an image to test the quantized TFLite model intended for ESP32.</p>
    
    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
        <p>Click to select an image or drag & drop here</p>
        <input type="file" id="fileInput" accept="image/jpeg, image/png, image/webp" onchange="previewImage(event)">
    </div>
    
    <img id="preview" src="" alt="Image Preview">
    
    <button class="btn" id="uploadBtn" onclick="uploadImage()" disabled>Classify Image</button>
    <div class="loading" id="loading">Running inference...</div>

    <div id="result">
        <h3 style="margin-top: 0; color: #fff; text-align: center;">Prediction Results</h3>
        <div class="stat-row">
            <span>Predicted Class:</span>
            <span class="stat-value" style="font-size: 1.2em;" id="predClass">-</span>
        </div>
        <div class="stat-row">
            <span>Confidence:</span>
            <span class="stat-value" id="predConf">-</span>
        </div>
        <div class="stat-row">
            <span>Inference Time:</span>
            <span class="stat-value" id="predTime">-</span>
        </div>
    </div>
</div>

<script>
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const preview = document.getElementById('preview');
    const resultDiv = document.getElementById('result');
    const loading = document.getElementById('loading');

    function previewImage(event) {
        const file = event.target.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
            uploadBtn.disabled = false;
            resultDiv.style.display = 'none';
        }
    }

    async function uploadImage() {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('image', file);

        uploadBtn.disabled = true;
        loading.style.display = 'block';
        resultDiv.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            document.getElementById('predClass').innerText = data.class;
            document.getElementById('predConf').innerText = (data.confidence * 100).toFixed(2) + '%';
            document.getElementById('predTime').innerText = data.time_ms.toFixed(2) + ' ms';
            
            resultDiv.style.display = 'block';
        } catch (error) {
            alert('Error during classification: ' + error);
        } finally {
            uploadBtn.disabled = false;
            loading.style.display = 'none';
        }
    }
</script>

</body>
</html>
"""

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_width, img_height))
    input_data = np.expand_dims(img, axis=0)
    
    # Check if the model expects quantized input (uint8) or float32
    if input_details[0]['dtype'] == np.float32:
        input_data = np.float32(input_data) / 255.0
    elif input_details[0]['dtype'] == np.uint8:
        input_data = np.uint8(input_data)
        
    return input_data

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    temp_path = "temp_upload.jpg"
    file.save(temp_path)
    
    try:
        input_data = prepare_image(temp_path)
        
        # Measure inference time
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        
        # De-quantize output if necessary
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
            
        predictions = output_data[0]
        max_idx = np.argmax(predictions)
        
        # Softmax if not already probabilities (simplistic check)
        if np.sum(predictions) > 1.1 or np.sum(predictions) < 0.9:
            # Just approximation, TFLite outputs might need softmax
            exp_p = np.exp(predictions - np.max(predictions))
            predictions = exp_p / np.sum(exp_p)
            
        confidence = float(predictions[max_idx])
        pred_class = LABELS[max_idx]
        inference_time_ms = (end_time - start_time) * 1000
        
        os.remove(temp_path)
        
        return jsonify({
            'class': pred_class,
            'confidence': confidence,
            'time_ms': inference_time_ms
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/esp32', methods=['POST'])
def esp32_predict():
    if 'image' not in request.files:
        return "ERROR: No image provided", 400
        
    file = request.files['image']
    temp_path = "temp_esp32_upload.jpg"
    file.save(temp_path)
    
    try:
        input_data = prepare_image(temp_path)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
            
        predictions = output_data[0]
        max_idx = np.argmax(predictions)
        
        os.remove(temp_path)
        
        # Return just the number as plain text for the ESP32!
        return str(max_idx)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "ERROR: " + str(e), 500

if __name__ == '__main__':
    print("Starting TFLite tester web app...")
    # Bind to 0.0.0.0 for cloud hosting
    app.run(host='0.0.0.0', port=5000, debug=False)
