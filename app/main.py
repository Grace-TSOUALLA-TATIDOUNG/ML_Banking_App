from fastapi import FastAPI
from app.schemas.input_data import ClientData

import uvicorn
import joblib
import pandas as pd


app = FastAPI()

pipeline = joblib.load("app/model/pipeline.pkl")


@app.get("/health")
def health_check():
    if pipeline is not None:
        return {
            "status": "healthy",
            "model_loaded": True
        }

    return {
        "status": "unhealthy",
        "model_loaded": False
    }


@app.post("/predict")
def predict(data: ClientData):

    df = pd.DataFrame([data.model_dump(by_alias=True)])

    prediction = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][0]

    return {
        "prediction": "yes" if prediction == 1 else "no",
        "probability": float(proba)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)