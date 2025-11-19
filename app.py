from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model at startup
MODEL_PATH = 'savedmodel.pth'
model = None

def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Load model when app starts
load_model()

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_flat = img_array.flatten() / 255.0  # Normalize to [0, 1]
    return img_flat.reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        # Preprocess image
        img_data = preprocess_image(file)
        # Make prediction
        prediction = model.predict(img_data)[0]
        prediction_proba = model.predict_proba(img_data)[0]
        # Get confidence
        confidence = float(np.max(prediction_proba) * 100)
        result = {
            'predicted_class': int(prediction),
            'confidence': f"{confidence:.2f}%",
            'message': f"Predicted Person: {prediction}"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    status = 'healthy' if model is not None else 'unhealthy'
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)