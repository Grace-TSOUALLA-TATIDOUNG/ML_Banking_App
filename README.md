# Bank Term Deposit Prediction System

Machine Learning API for Bank Marketing Campaign Prediction using FastAPI, Streamlit, Docker, and CI/CD.

---

## Project Overview

This project is an end-to-end Machine Learning application built on a real-world banking marketing dataset. The objective is to predict whether a client is likely to subscribe to a term deposit based on customer demographics, campaign information, and socio-economic indicators.

The project covers the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, deployment, and user interaction.

The final solution consists of:

* A trained Machine Learning model
* A FastAPI REST API exposing prediction endpoints
* A Streamlit front-end application
* Docker containerization
* CI/CD automation with GitHub Actions
* Cloud deployment on Render

The application demonstrates how a Machine Learning model can be transformed into a production-oriented solution accessible through both an API and a user-friendly interface.

---

## Architecture Overview

```text
| Machine Learning Workflow | Application Architecture |
|--------------------------|---------------------------|
| 📊 Dataset | 👤 End User |
| ⬇️ | ⬇️ |
| 🧹 Data Cleaning & Preprocessing | 🖥️ Streamlit Front-End |
| ⬇️ | ⬇️ |
| ⚙️ Feature Engineering | 🚀 FastAPI REST API |
| ⬇️ | ⬇️ |
| 🤖 Model Training & Evaluation | 💾 Serialized ML Model |
| ⬇️ | ⬇️ |
| 💾 Serialized ML Model | 📋 Prediction Result |
```

---

## Dataset

This project uses the Bank Marketing Dataset available on Kaggle:

https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

### Dataset Characteristics

* More than 41,000 customer records
* Demographic information
* Marketing campaign data
* Socio-economic indicators
* Binary classification target

### Target Variable

| Value | Description                                |
| ----- | ------------------------------------------ |
| yes   | Client subscribed to a term deposit        |
| no    | Client did not subscribe to a term deposit |

---

## Machine Learning Workflow

### Data Preparation

* Data cleaning and preprocessing
* Feature encoding
* Data type optimization
* Feature engineering
* Train-test split

### Exploratory Data Analysis

* Customer profile analysis
* Target distribution analysis
* Campaign effectiveness analysis
* Feature relationship exploration

### Model Development

Several Machine Learning algorithms were evaluated and compared, including:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The best-performing model was selected and integrated into a reusable prediction pipeline.

### Model Evaluation

Models were assessed using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross-Validation

---

## API Development

The trained model is served through a REST API built with FastAPI.

### Main Endpoint

```http
POST /predict
```

### Example Request

```json
{
  "age": 42,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "duration": 300,
  "campaign": 2,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp.var.rate": 1.1,
  "cons.price.idx": 93.994,
  "cons.conf.idx": -36.4,
  "euribor3m": 4.857,
  "nr.employed": 5191
}
```

### Example Response

```json
{
  "prediction": "yes",
  "probability": 0.536
}
```

The API returns:

* The predicted class
* The probability associated with the prediction

---

## Interactive API Documentation

The API can be tested directly through the automatically generated Swagger UI.

### API Documentation

```text
https://bank-marketing-prediction-api.onrender.com/docs
```

### API Base URL

```text
https://bank-marketing-prediction-api.onrender.com
```

---

## Streamlit Front-End

The project also includes a Streamlit application that provides a simple user interface for interacting with the deployed API.

### Features

* User-friendly input form
* Real-time prediction requests
* Prediction result display
* Prediction probability display
* API integration without requiring direct API usage

### Run the Front-End

```bash
streamlit run frontend/app.py
```

---

## Project Structure

```text
ML_Banking_App/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── model/
│   └── artifacts/
│
├── frontend/
│   └── app.py
│
├── notebooks/
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
│
└── data/
```

---

## Running the Project Locally

### Clone the Repository

```bash
git clone https://github.com/<your-username>/ML_Banking_App.git
cd ML_Banking_App
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

### Build the Docker Image

```bash
docker build -t bank-marketing-api .
```

### Run the Container

```bash
docker run -p 8000:8000 bank-marketing-api
```

---

## CI/CD Pipeline

The project includes a CI/CD pipeline implemented with GitHub Actions.

The workflow automatically:

1. Installs project dependencies
2. Runs automated tests
3. Builds the Docker image
4. Pushes the image to GitHub Container Registry (GHCR)
5. Prepares the application for deployment

This setup simulates a production-oriented Machine Learning deployment workflow.

---

## Deployment

### API Deployment

The FastAPI application is deployed on Render and publicly accessible.

```text
https://bank-marketing-prediction-api.onrender.com
```

### Container Registry

Docker images are automatically published to GitHub Container Registry (GHCR) through GitHub Actions.

---

## Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Predictive Analytics
* Machine Learning Model Development
* Model Evaluation and Optimization
* Pipeline Creation and Model Serialization
* REST API Development with FastAPI
* Front-End Development with Streamlit
* Docker Containerization
* Automated Testing
* Git and GitHub Version Control
* CI/CD with GitHub Actions
* Cloud Deployment
* MLOps Fundamentals

---

## Future Improvements

* Streamlit cloud deployment
* Model monitoring and logging
* Automated model retraining
* Experiment tracking
* Enhanced MLOps workflows

---

## Author

**Grace TSOUALLA TATIDOUNG**

Data Scientist & Data Analyst

Passionate about transforming data into actionable insights and building end-to-end Machine Learning solutions that create measurable business value.
