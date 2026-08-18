import os
import time
import numpy as np
import datetime
import shutil
from PIL import Image
from flask import Flask, request

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
    # A simple ping endpoint for UptimeRobot to keep the server awake
    return "ESP32 Waste Sorting API is running!", 200

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
        
        # Save the image instead of deleting it
        captures_dir = "captures"
        os.makedirs(captures_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        predicted_label = LABELS[max_idx]
        new_filename = f"{timestamp}_{predicted_label}.jpg"
        new_path = os.path.join(captures_dir, new_filename)
        shutil.move(temp_path, new_path)
        print(f"Saved capture to: {new_path}")
        
        # Return just the number as plain text for the ESP32!
        return str(max_idx)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "ERROR: " + str(e), 500

from flask import send_from_directory

@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory('captures', filename)

@app.route('/captures', methods=['GET'])
def view_captures():
    captures_dir = "captures"
    if not os.path.exists(captures_dir):
        return "No captures yet! Use the ESP32 to take some photos."
    
    files = sorted(os.listdir(captures_dir), reverse=True)
    if not files:
        return "No captures found in the directory."
        
    html = "<h2>ESP32 Waste Captures</h2><div style='display:flex; flex-wrap:wrap; gap:10px;'>"
    for f in files:
        if f.endswith('.jpg'):
            html += f"<div style='border:1px solid #ccc; padding:10px; border-radius:5px;'>"
            html += f"<img src='/captures/{f}' style='width:320px; height:240px; display:block;' />"
            html += f"<p style='text-align:center; font-family:sans-serif;'>{f}</p>"
            html += "</div>"
    html += "</div>"
    return html

if __name__ == '__main__':
    print("Starting ESP32 API...")
    app.run(host='0.0.0.0', port=5000, debug=False)
