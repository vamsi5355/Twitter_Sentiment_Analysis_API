Twitter Sentiment Analysis API (BERT – MLOps Project)
📌 Project Overview
This project implements a complete sentiment analysis system using a pre-trained BERT model and demonstrates a production-style MLOps pipeline.
The system covers data preprocessing, model fine-tuning, evaluation, REST API deployment, batch inference, and a simple UI.
The goal of this project is not only to build a machine learning model, but to show how ML models are trained, evaluated, served, and tested in real-world applications such as:
Social media monitoring
Customer feedback analysis
Brand sentiment tracking
🧠 Model & Technology Stack
Machine Learning
Model: BERT (bert-base-uncased)
Task: Binary Sentiment Classification (positive / negative)
Framework: Hugging Face Transformers
Training: Fine-tuning on IMDB dataset
Backend & Serving
API Framework: FastAPI
Inference: PyTorch
Batch Prediction: Python script
UI
Framework: Streamlit
MLOps & Tooling
Experiment metrics logging
Model artifact persistence
Environment-based configuration
Docker & Docker Compose (optional deployment)
Twitter Sentiment Analysis API/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
├── model_output/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── results/
│   ├── metrics.json
│   └── run_summary.json
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── batch_predict.py
├── src/
│   ├── api.py
│   └── ui.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
├── .env.example
└── README.md
Setup Instructions
1️⃣ Prerequisites
Make sure you have:
Python 3.10+
pip
(Optional) Docker & Docker Compose
Create and activate a virtual environment (recommended):
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
How to Test the Project
3️⃣ Data Preprocessing
This step downloads the IMDB dataset, cleans the text, and creates train/test splits.
python scripts/preprocess.py
Model Training
Fine-tunes a pre-trained BERT model and evaluates it.
python scripts/train.py
model_output/
 ├── config.json
 ├── pytorch_model.bin
 ├── tokenizer_config.json
 └── vocab.txt
results/
 ├── metrics.json
 └── run_summary.json
Run the API Server
Start the FastAPI backend:
uvicorn src.api:app --reload --port 8000
Test API Endpoints
🔹 Health Check
Invoke-RestMethod http://localhost:8000/health
