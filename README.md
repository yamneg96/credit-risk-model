# 🏦 Credit Risk Probability Model – Final Submission

## 🚀 Overview
This is a full-stack ML project to predict credit risk probability for **Bati Bank's** buy-now-pay-later service. It transforms customer behavioral transaction data into actionable insights for **risk-based decision-making**, while adhering to **Basel II Accord** compliance.

Key outputs:
- Probability-based risk scores
- Proxy-based default classification
- Model deployed as a REST API (FastAPI)
- MLOps practices via GitHub Actions, Docker, and MLflow

---

## 🔍 Business Understanding

- **Challenge:** No default labels, so proxy defaults are derived from behavioral signals using RFM (Recency, Frequency, Monetary).
- **Goal:** Predict likelihood of customer default with interpretable + performant models.
- **Basel II:** Drives the need for model explainability, audit trails, and classification fairness.

---

## 📁 Folder Structure

```
credit-risk-model/
├── .github/workflows/
│   └── ci.yml                   # CI/CD pipeline
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── mlruns/                  # MLflow experiments
├── notebooks/
│   ├── 1.0-eda.ipynb
│   └── 2.0-model-eval.ipynb     # Evaluation & comparison
├── reports/
│   ├── interim_report.pdf
│   └── final_report.pdf
├── src/
│   ├── data_processing.py
│   ├── train.py                 # Model training script
│   ├── predict.py               # Inference script
│   └── api/
│       ├── main.py              # FastAPI app
│       └── pydantic_models.py   # Input schema definitions
├── tests/
│   └── test_data_processing.py
├── Dockerfile                   # For containerization
├── docker-compose.yml           # Optional for orchestration
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

```bash
# Clone repo & setup virtualenv
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/data_processing.py

# Train & track model
python src/train.py

# Run FastAPI
uvicorn src.api.main:app --reload

# Run unit tests
pytest tests/

# Build Docker image
docker build -t credit-risk-api .

# Run app with Docker
docker run -p 8000:8000 credit-risk-api
```

---

## 🧪 Models Used

- Logistic Regression (with WoE)
- Random Forest
- Gradient Boosting (XGBoost)
- Evaluation: Accuracy, F1, Precision, ROC-AUC

---

## 📦 Key Technologies

- **scikit-learn** – modeling & pipelines
- **MLflow** – experiment tracking
- **FastAPI** – model deployment
- **Docker** – containerization
- **GitHub Actions** – CI/CD
- **pytest** – unit testing

---

## 📈 Final Deliverables

- `final_report.pdf` (blog/report style)
- Source code and tests
- REST API server via FastAPI
- MLflow tracking for model comparison
- CI pipeline: lint + test (GitHub Actions)
