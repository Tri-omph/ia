from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model once when the server starts
onnx_model_path = "version/v1/IA-v1.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image_onnx(image_bytes, input_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(input_size)  # input_size = (height, width), resize expects (width, height)
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize between 0 and 1

    # Normalize based on ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Change format to (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the image
    img_bytes = file.read()

    # Preprocess the image
    img_input = preprocess_image_onnx(img_bytes)

    # Run the prediction
    inputs = {input_name: img_input}
    outputs = session.run([output_name], inputs)

    # Find the predicted class
    predicted_class_idx = np.argmax(outputs[0])
    classes = ["PILE",
            "CARTON",
            "VERRE",
            "MÉTAL",
            "PAPIER",
            "PLASTIQUE",
            "TEXTILE",
            "ORDURES",]
    predicted_class_name = classes[predicted_class_idx]

    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    #app.run(debug=True)
