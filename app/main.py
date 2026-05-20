from fastapi import FastAPI, HTTPException
from app.schemas.input_data import ClientData
from app.schemas.output_data import PredictionResponse

import uvicorn
import joblib
import pandas as pd


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API to predict if a Client will subscribe to a term deposit",
    version="1.0.0"
)

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


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded"
        )

    try:
        df = pd.DataFrame([data.model_dump(by_alias=True)])

        pred = pipeline.predict(df)[0]
        proba = pipeline.predict_proba(df)[0][1]

        return PredictionResponse(
            prediction="yes" if pred == 1 else "no",
            probability=float(proba)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)