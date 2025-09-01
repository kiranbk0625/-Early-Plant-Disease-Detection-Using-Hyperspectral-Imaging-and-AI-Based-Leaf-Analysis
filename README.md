# Plant Disease Detection

This project is a deep learning-based application for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN).
 Early Plant Disease Detection Using Hyperspectral  Imaging and AI-Based Leaf Analysis

It is built with TensorFlow/Keras and trained on the -- PlantVillage dataset(https://www.kaggle.com/datasets/emmarex/plantdisease?utm_source=chatgpt.com)


Project Structure

plant-disease-detection/
│── model_app.py        # Main script for training & prediction
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── saved_model/        # Trained model
│── dataset/            # Dataset folder (PlantVillage images)
│   ├── train/
│   ├── val/
│   └── test/


⚡ Features

Detects plant diseases from leaf images.

Trained on 38 classes from the PlantVillage dataset.

Shows Top-3 predictions with probabilities.

Can be extended to real-time applications using Flask/FastAPI + React.



🚀 Getting Started
1️⃣ Clone the Repository

git clone https://github.com/kiranbk0625/plant-disease-detection.git
cd plant-disease-detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Download Dataset

import kagglehub
path = kagglehub.dataset_download("emmarex/plantdisease")
print("Dataset downloaded to:", path)


🛠 Tech Stack

Python 3.x

TensorFlow / Keras

NumPy, Matplotlib, Pillow

KaggleHub API (for dataset)

🔮 Future Improvements

Add Flask/FastAPI API for serving predictions.

Create React web dashboard for farmers.

Deploy model to cloud (AWS, GCP, or Heroku).

📜 License

This project is open-source and available under the MIT License.

✨ Contributions are welcome! Fork, star ⭐, and submit a PR if you'd like to improve this project.
