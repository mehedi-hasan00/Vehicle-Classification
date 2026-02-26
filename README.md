# 🚗✈️🚢 Vehicle Type Classification

A Deep Learning based **Image Classification** project trained to recognize different types of vehicles. This model uses **EfficientNetV2B0 (Transfer Learning)** to extract visual features and classify images with high accuracy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

🌐 **Live App:** [https://vehicle-classification.streamlit.app/](https://vehicle-classification.streamlit.app/)

---

## 🚀 Live Demo

🔗 **Streamlit App:** [https://vehicle-classification.streamlit.app/](https://vehicle-classification.streamlit.app/)

Run the Streamlit app to upload an image and get instant vehicle classification.

**Input:** `Upload an image of a car, plane, or ship...`

**AI Output (Example):** `This image most likely belongs to **Cars** (Confidence: 99.45%)`

---

## 🧠 Project Overview

The goal of this project is to build a robust computer vision model capable of categorizing **7 different vehicle types**.

By utilizing transfer learning with EfficientNetV2B0, the model learns:
* **Visual Features** – identifies diverse vehicle classes accurately
* **Shape & Texture** – recognizes structural patterns of transport
* **Domain Adaptation** – distinguishes between land, air, and water vehicles

**Supported Classes:** Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains.

---

## 🛠️ Model Architecture

Built using **Keras / TensorFlow**:
1. **Pretrained Base Model** – EfficientNetV2B0 (ImageNet weights)
2. **Global Average Pooling Layer** – reduces spatial dimensions
3. **Batch Normalization Layer** – stabilizes and speeds up training
4. **Dense Layer** – with ReLU activation for feature learning
5. **Dropout Layer** – (reduces overfitting)
6. **Dense Output Layer** – Softmax activation (7 classes)

### Training Configuration:
* **Image Size:** 224 x 224
* **Batch Size:** 32
* **Channels:** 3 (RGB)
* **Loss Function:** `sparse_categorical_crossentropy`
* **Optimizer:** Adam
* **Evaluation Metrics:** Accuracy


---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mehedi-hasan00/Vehicle-Classification.git
cd Vehicle-Classification
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```text
├── classification.ipynb         # Model training & evaluation notebook
├── app.py                       # Streamlit web app
├── models/
│   └── model.h5                 # Trained EfficientNetV2B0 model
├── class_names.json             # JSON file containing class labels
├── pyproject.toml               # Project metadata and configuration
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

---

## 📝 requirements.txt
Keep this file in the repository to ensure smooth setup and deployment on Streamlit Cloud.

```text
streamlit
tensorflow
numpy
Pillow
opencv-python
matplotlib
```

---

## 📊 Performance

### 📂 Dataset Overview
* **Total Images Found:** 5,588 images
* **Total Classes:** 7

### 📉 Dataset Split (Batch Size: 32)
* **Training Set:** 140 Batches (Approx. 4,480 images)
* **Validation Set:** 17 Batches (Approx. 544 images)
* **Test Set:** 18 Batches (Approx. 576 images)

### 📈 Model Accuracy
* Training Accuracy: ~97–98%
* Validation Accuracy: ~99–99.8%
* Test Accuracy: ~99–99.8%

The model generalizes well across varied backgrounds, lighting conditions, and angles.

---

## 👤 Author
**Mehedi Hasan**
* 🔗 LinkedIn: [https://www.linkedin.com/in/mehedi-hasan-094855388/](https://www.linkedin.com/in/mehedi-hasan-094855388/)
* 🔗 Kaggle: [https://www.kaggle.com/mehedi71](https://www.kaggle.com/mehedi71)
