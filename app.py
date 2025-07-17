from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = load_model('crop_disease_model.h5')

# Load class labels
with open('class_labels.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# Treatment suggestions
treatment_db = {
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds and copper-based bactericides.",
    "Pepper__bell___healthy": "Healthy crop. Continue regular monitoring.",
    "Potato___Early_blight": "Apply chlorothalonil-based fungicides and rotate crops.",
    "Potato___Late_blight": "Apply fungicides and avoid overhead watering.",
    "Potato___healthy": "Healthy crop. Maintain good irrigation.",
    "Tomato_Bacterial_spot": "Use copper sprays and remove infected plants.",
    "Tomato_Early_blight": "Use copper-based fungicides. Remove infected leaves.",
    "Tomato_Late_blight": "Apply fungicides and ensure good drainage.",
    "Tomato_Leaf_Mold": "Ensure ventilation and apply fungicides if necessary.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides and maintain humidity.",
    "Tomato__Target_Spot": "Apply fungicides and maintain crop rotation.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato__Tomato_mosaic_virus": "Avoid handling when plants are wet and use resistant varieties.",
    "Tomato_healthy": "No treatment needed.",
}


def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred))
    return pred_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        disease, confidence = predict_disease(filepath)
        treatment = treatment_db.get(disease, "Consult an agricultural expert.")

        return jsonify({
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "treatment": treatment
        })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
