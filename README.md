
# 🌿 Crop Disease Detector AI (Flask + TensorFlow)

This is an AI-powered web application that detects plant leaf diseases using deep learning. Users can upload a leaf image and receive predictions along with possible treatment suggestions.

---

## 🚀 Features

- Trained on PlantVillage dataset (16 crop diseases + healthy classes)
- ResNet50 deep learning model
- Real-time image prediction via a Flask web app
- Upload leaf image to get:
  - Predicted disease name
  - Confidence score
  - Treatment suggestion

---

## 🧠 Trained On

- Dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes (examples):
  - `Tomato_Early_blight`
  - `Tomato_Late_blight`
  - `Tomato_Leaf_Mold`
  - `Potato_healthy`
  - `Pepper__bell___Bacterial_spot`
  - ...and more

---

## 🖼 Example

Upload image ➜ Model predicts disease ➜ Displays treatment advice

---

## 🛠 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/MemoonaHassan/crop-disease-detector.git
cd crop-disease-detector
````

### 2. (Optional) Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional if `crop_disease_model.h5` already exists)

```bash
python data_preprocessing.py
python model_training.py
```

### 5. Run the Flask App

```bash
python app.py
```

Then open your browser and go to:
`http://127.0.0.1:5000/`

---

## 🧪 File Structure

```
crop-disease-detector/
│
├── app.py                # Flask web application
├── data_preprocessing.py # Image preprocessing & dataset loader
├── model_training.py     # Training script (ResNet50)
├── class_labels.txt      # List of disease class names
├── templates/
│   └── index.html        # Frontend interface
├── uploads/              # Temporary uploaded images
├── dataset/              # Training & validation dataset folders
├── crop_disease_model.h5 # Trained model (if available)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 💡 Treatment Suggestion Logic

Treatment is suggested based on the predicted class:

```python
treatment_db = {
    "Tomato_Early_blight": "Use copper-based fungicides and remove infected leaves.",
    "Potato_Late_blight": "Apply fungicides and avoid overhead watering.",
    "Tomato_healthy": "No treatment needed.",
}
```

---


## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and submit improvements.

---

## 📄 License

MIT License – do whatever you want 😄

---

