# Australia Rain Prediction 🌧️

A machine learning project that predicts whether it will rain tomorrow in Australia, using historical weather data and an Artificial Neural Network (ANN).  
The project also includes a simple web application and Docker setup for deployment.

---

## 📌 Table of Contents

- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Docker Setup](#docker-setup)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Motivation

Predicting rainfall is an important task for meteorology, agriculture, and disaster management.  
This project demonstrates how deep learning can be applied to weather prediction using Australian climate data.

---

## 📂 Project Structure

Australia-Rain-Prediction/
├── Dataset/RAIN DATASETS/ # Weather data files
├── k8s/ # Kubernetes configuration (optional)
├── Australia_Rain_Prediction_ANN.ipynb # Jupyter notebook with training & analysis
├── app.py # Web / API server (Flask or similar)
├── model.h5 # Trained ANN model
├── Dockerfile # Docker build file
└── requirements.txt # Python dependencies

---

## ✨ Features

- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- ANN-based prediction model
- Flask API / web interface for predictions
- Docker containerization for deployment

---

## 📊 Dataset

The dataset is located in the `Dataset/RAIN DATASETS/` folder and contains historical Australian weather observations.  
Features include temperature, humidity, wind speed, cloud cover, and other atmospheric data.

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/chintan0513/Australia-Rain-Prediction.git
   cd Australia-Rain-Prediction
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

▶️ Usage
Train & Explore the Model

Run the Jupyter notebook:

jupyter notebook Australia_Rain_Prediction_ANN.ipynb

Run the API / Web App

Start the Flask app:

python app.py

By default, the API will run at http://127.0.0.1:5000/.

🧠 Model Architecture

Type: Artificial Neural Network (ANN)

Input: Weather attributes (temperature, humidity, wind, etc.)

Output: Binary classification → Rain / No Rain

Loss function: Binary cross-entropy

Metrics: Accuracy, precision, recall, F1-score

🐳 Docker Setup

Build the Docker image:

docker build -t australia-rain-prediction .

Run the container:

docker run -p 5000:5000 australia-rain-prediction

Now the API is available inside the container at port 5000
