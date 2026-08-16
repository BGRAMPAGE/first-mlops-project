# First MLOps Project – Learning the MLOps Lifecycle

> **This is not a project built to achieve the best house price prediction accuracy.** The California Housing dataset is used as a simple baseline so the focus can remain on learning and implementing the core concepts of **MLOps**—from training a model to tracking experiments, versioning, serving, containerizing, and deploying it.

## Project Goal

The objective of this project is to understand how an ML model moves through a production-like workflow rather than building a highly optimized machine learning model.

This project demonstrates:

* Model training with Scikit-learn
* Experiment tracking using MLflow
* Model versioning concepts
* REST API serving with FastAPI
* Docker containerization
* Cloud deployment using Render

---

## Tech Stack

| Category            | Tools        |
| ------------------- | ------------ |
| Language            | Python 3.12  |
| Machine Learning    | Scikit-learn |
| Experiment Tracking | MLflow       |
| API                 | FastAPI      |
| Validation          | Pydantic     |
| Containerization    | Docker       |
| Deployment          | Render       |
| Testing             | Python       |

---

## MLOps Workflow Implemented

<svg viewBox="0 0 900 120" width="100%" role="img" aria-label="Implemented MLOps workflow">
  <rect x="20" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="65" y="60" text-anchor="middle" font-size="14">Data</text>
  <path d="M110 55 L155 55" stroke="currentColor" strokeWidth="2"/><polygon points="155,55 143,49 143,61" fill="currentColor"/>
  <rect x="155" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="200" y="60" text-anchor="middle" font-size="14">Train</text>
  <path d="M245 55 L290 55" stroke="currentColor" strokeWidth="2"/><polygon points="290,55 278,49 278,61" fill="currentColor"/>
  <rect x="290" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="335" y="60" text-anchor="middle" font-size="14">MLflow</text>
  <path d="M380 55 L425 55" stroke="currentColor" strokeWidth="2"/><polygon points="425,55 413,49 413,61" fill="currentColor"/>
  <rect x="425" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="470" y="60" text-anchor="middle" font-size="14">Registry</text>
  <path d="M515 55 L560 55" stroke="currentColor" strokeWidth="2"/><polygon points="560,55 548,49 548,61" fill="currentColor"/>
  <rect x="560" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="605" y="60" text-anchor="middle" font-size="14">API</text>
  <path d="M650 55 L695 55" stroke="currentColor" strokeWidth="2"/><polygon points="695,55 683,49 683,61" fill="currentColor"/>
  <rect x="695" y="30" width="90" height="50" rx="10" fill="#d1fae5" stroke="#10b981"/>
  <text x="740" y="60" text-anchor="middle" font-size="14">Docker</text>
  <path d="M785 55 L830 55" stroke="currentColor" strokeWidth="2"/><polygon points="830,55 818,49 818,61" fill="currentColor"/>
  <rect x="830" y="30" width="50" height="50" rx="10" fill="#f3f4f6" stroke="#9ca3af"/>
  <text x="855" y="60" text-anchor="middle" font-size="14">Ops</text>
</svg>

The project follows this workflow:

1. Train a Random Forest model.
2. Evaluate performance using RMSE and R².
3. Track experiments with MLflow.
4. Log parameters, metrics, and artifacts.
5. Register model versions.
6. Serve predictions through FastAPI.
7. Package everything using Docker.
8. Deploy the API on Render.

---

## Features Implemented

### Model Training

* California Housing Dataset
* Random Forest Regressor
* Performance evaluation

### MLflow Experiment Tracking

* Experiment creation
* Parameter logging
* Metric logging
* Multiple training runs
* Model registration
* Artifact storage

Example tracked artifacts include:

* Feature importance plots
* Training reports
* Model versions

### FastAPI Prediction API

The trained model is exposed through a REST API.

Endpoints include:

* `/`
* `/predict`
* `/docs`

### Docker Support

The application is fully containerized.

```bash
docker build -t house-api .
docker run -p 8000:8000 house-api
```

### Render Deployment

The Docker container is deployed as a cloud-hosted API using Render.

---

## Project Structure

```text
First_ml_project/
├── main.py
├── model_house.py
├── test_model.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── .dockerignore
```

Generated files such as:

* `mlruns/`
* `mlflow.db`
* model artifacts
* training reports

are intentionally excluded from Git using `.gitignore`.

---

## What I Learned

This project helped me understand practical MLOps concepts including:

* Experiment tracking
* Model versioning
* API deployment
* Docker packaging
* Cloud deployment workflows
* Repository hygiene for ML projects

More importantly, it showed how a machine learning model progresses from a local experiment into a deployable service.

---

## Planned Next Steps

This repository is intentionally designed as a foundation for more advanced MLOps practices.

Future additions include:

* GitHub Actions (CI/CD)
* Prometheus monitoring
* Grafana dashboards
* Data drift detection (Evidently AI)
* Airflow/Prefect pipelines
* Kubernetes deployment
* Production model monitoring

---

## Why This Repository Exists

Many beginner ML projects stop after training a model.

This repository focuses on the next stage—**the operational side of machine learning**—by exploring the tools and workflows used to manage models throughout their lifecycle.

The simple model is a deliberate choice so the emphasis stays on building practical MLOps skills rather than chasing benchmark accuracy.
