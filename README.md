# 🇺🇸 US Visa Approval Prediction — MLOps Project

A **production-grade, end-to-end MLOps pipeline** that predicts whether a US visa application will be **Certified** or **Denied**, based on applicant and employer information. The main goal was to build a proper end-to-end machine learning pipeline — not just a notebook experiment, but something that can actually run in a structured, repeatable way.

---

## 🏗️ Architecture

```
MongoDB Atlas
     ↓
[1] DataIngestion      → exports CSV, 80/20 split
     ↓
[2] DataValidation     → schema check + Evidently drift detection
     ↓
[3] DataTransformation → feature engineering, encoding, SMOTEENN (train only)
     ↓
[4] ModelTrainer       → KNN / RandomForest / GradientBoosting + MLflow tracking
     ↓
[5] ModelEvaluation    → compares vs. best model (threshold: ΔF1 ≥ 0.02)
     ↓
[6] ModelPusher        → saves to saved_models/ + optional S3 upload
     ↓
FastAPI App            → /predict (form) | /api/v1/predict (JSON) | /monitor
```

---

## 📂 Project Structure

```
visa project MLOPS/
├── USvisa/
│   ├── components/           # 6 pipeline stage components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py      # + MLflow tracking
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py       # saves to saved_models/ + S3
│   ├── pipeline/
│   │   ├── training_pipeline.py  # orchestrates all 6 stages
│   │   └── prediction_pipeline.py
│   ├── entity/
│   │   ├── config.py             # typed dataclass configs
│   │   ├── artifact.py           # typed dataclass artifacts
│   │   └── estimator.py          # USvisaModel + TargetValueMapping
│   ├── constants/                # centralized constants
│   ├── data_access/              # MongoDB → DataFrame
│   ├── configuration/            # MongoDB connection singleton
│   ├── utils/                    # YAML, pickle, numpy helpers
│   ├── exception/                # custom exception (file + line)
│   └── logger/                   # timestamped file logging
├── tests/                        # pytest unit tests
│   ├── test_utils.py
│   ├── test_entity.py
│   └── test_pipeline.py
├── config/
│   └── schema.yaml               # feature schema, drop columns, encoders
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_eng.ipynb
│   └── MONGO.ipynb
├── .github/workflows/ci.yml      # lint → tests → docker build → train
├── app.py                        # FastAPI app
├── templates/index.html          # prediction UI
├── demo.py                       # run training pipeline
├── Dockerfile
├── Makefile
├── requirements.txt              # production deps
├── requirements-dev.txt          # dev/test/notebook deps
└── .env.example                  # env variable template
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (connection string in `.env`)
- Docker (optional, for containerized deployment)

### 1 — Clone and setup

```bash
git clone https://github.com/RajatVerma01/VISA_PREDICTION_MLOPS.git
cd "visa project MLOPS"

# Copy environment template
cp .env.example .env
# → Fill in MONGODB_CONNECTION_STRING in .env
```

### 2 — Install dependencies

```bash
make install          # production deps only
make install-dev      # + dev / test / notebook deps
```

### 3 — Train the model

```bash
make train
# Artifacts saved under artifact/<timestamp>/
# Accepted model copied to saved_models/model.pkl
```

### 4 — Start the web app

```bash
make serve
# → http://localhost:8000        (prediction form)
# → http://localhost:8000/docs   (Swagger API docs)
# → http://localhost:8000/health (liveness probe)
# → http://localhost:8000/monitor (drift report)
```

### 5 — View MLflow experiment dashboard

```bash
make mlflow
# → http://localhost:5000
```

---

## 🐳 Docker

```bash
make docker-build   # builds usvisa-predictor:latest
make docker-run     # runs on port 8000 with .env
```

Or manually:
```bash
docker build -t usvisa-predictor .
docker run -p 8000:8000 --env-file .env usvisa-predictor
```

---

## 🧪 Testing

```bash
make test    # runs pytest with coverage report
make lint    # runs flake8
```

---

## 🔁 CI/CD (GitHub Actions)

| Trigger | Jobs |
|---|---|
| Push / PR to `main` | Lint → Tests → Docker Build+Push |
| Manual (`workflow_dispatch`) | + Full Training Pipeline |

**Required GitHub Secrets:**
| Secret | Required |
|---|---|
| `DOCKERHUB_USERNAME` | For Docker push |
| `DOCKERHUB_TOKEN` | For Docker push |
| `MONGODB_CONNECTION_STRING` | For training job |
| `MODEL_BUCKET_NAME` | Optional (S3 model registry) |
| `AWS_ACCESS_KEY_ID` | Optional (S3) |
| `AWS_SECRET_ACCESS_KEY` | Optional (S3) |
| `AWS_DEFAULT_REGION` | Optional (S3) |

---

## 📊 API Reference

### `POST /api/v1/predict` (JSON)
```json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": 500,
  "yr_of_estab": 1995,
  "region_of_employment": "Northeast",
  "prevailing_wage": 75000.0,
  "unit_of_wage": "Year",
  "full_time_position": "Y"
}
```
**Response:**
```json
{
  "prediction": "Certified",
  "confidence": "high",
  "model_loaded_from": "saved_models/model.pkl"
}
```

### `GET /health`
```json
{"status": "ok", "model": "loaded", "model_path": "saved_models/model.pkl"}
```

---

## 🗃️ Dataset Features

| Column | Type | Role |
|---|---|---|
| `continent` | category | OneHot encoded |
| `education_of_employee` | category | Ordinal encoded |
| `has_job_experience` | Y/N | Ordinal encoded |
| `requires_job_training` | Y/N | Ordinal encoded |
| `no_of_employees` | int | PowerTransform + Scale |
| `yr_of_estab` | int | → `company_age` (engineered) |
| `region_of_employment` | category | OneHot encoded |
| `prevailing_wage` | float | StandardScaler |
| `unit_of_wage` | category | OneHot encoded |
| `full_time_position` | Y/N | Ordinal encoded |
| `case_status` | **Target** | Certified=0 / Denied=1 |

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| ML | scikit-learn, XGBoost, CatBoost |
| Imbalance | SMOTEENN (train-only) |
| Drift Detection | Evidently ≥ 0.4 |
| Experiment Tracking | MLflow |
| Data Storage | MongoDB Atlas |
| Serialization | dill |
| Web API | FastAPI + Uvicorn |
| Container | Docker |
| CI/CD | GitHub Actions |
| Environment | python-dotenv |

---

*Built by Rajat Verma*
