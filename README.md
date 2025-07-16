# 🛡️ Phishing Website Detection using Machine Learning

A machine learning-powered web application to detect phishing websites based on URL-related features. This project uses classical ML models to classify a website as **Legitimate**, **Phishing** and deploys it using FastAPI.

---

## 📂 Project Structure


---

## 📊 Dataset Overview

The dataset contains **30 features** extracted from URLs, plus a target label `Result`:

| Label        | Meaning         |
|--------------|-----------------|
| `0`          | Legitimate      |
| `1`          | Phishing        |

### Example Features:
- `having_IP_Address`
- `URL_Length`
- `Shortining_Service`
- `SSLfinal_State`
- `Abnormal_URL`
- `Google_Index`
- `Result` (Target)

---

## 🎯 Objective

Build and deploy a robust model that accurately detects phishing websites based on their structural and behavioral URL features.

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Clean and validate raw data
- Encode categorical features
- Split into train/test sets

### 2. Exploratory Data Analysis (EDA)
- Feature distribution and correlation
- Class imbalance analysis

### 3. Model Training
- Algorithms used: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost
- Hyperparameter tuning
- Performance logging with MLflow and Dagshub

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Tools: Confusion Matrix, ROC Curve

### 5. Model Deployment (FastAPI)
- REST API for batch and individual predictions
- Swagger/OpenAPI docs enabled
- Upload CSVs or input via form

---

## 🚀 FastAPI Endpoints

| Method | Route           | Description                        |
|--------|------------------|------------------------------------|
| `GET`  | `/train`         | Trigger model training pipeline    |
| `POST` | `/predict`       | Upload CSV for batch predictions   |
| `POST` | `/predict-single`| Input single record via form       |
| `GET`  | `/docs`          | Access Swagger UI documentation    |

---

## 🌐 Web Interface Features

- Upload CSV file for **batch predictions**
- Input form for **single prediction**
- Results displayed in a table
- **Download** prediction output as `.csv`

---

## 📥 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/visteen192/network-security.git
cd network-security

```
## Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

## Install Dependencies
```bash
pip install -r requirements.txt
```