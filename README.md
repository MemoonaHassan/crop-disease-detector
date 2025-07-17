# 🌿 Crop Disease Detector AI (Flask + TensorFlow)

This is an AI-powered web application that detects plant leaf diseases using deep learning. Users can upload a leaf image and receive predictions along with possible treatment suggestions.

## 🚀 Features

- Trained on PlantVillage dataset (16 crop diseases + healthy classes)
- ResNet50 deep learning model
- Real-time image prediction via a Flask web app
- Upload leaf image to get:
  - Predicted disease name
  - Confidence score
  - Treatment suggestion

## 🧠 Trained On

- Dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes (examples):
  - Tomato_Early_blight
  - Tomato_Late_blight
  - Tomato_Leaf_Mold
  - Potato_healthy
  - Pepper__bell___Bacterial_spot
  - ...and more

## 🖼 Example

Upload image ➜ Model predicts disease ➜ Displays treatment advice

## 🛠 How to Run Locally
# 1. Clone the repository
git clone https://github.com/MemoonaHassan/crop-disease-detector.git
cd crop-disease-detector

# 2. (Optional but recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install the required Python packages
pip install -r requirements.txt

# 4. (Optional) Preprocess and train the model if not already available
python data_preprocessing.py
python model_training.py

# 5. Run the Flask web application
python app.py


