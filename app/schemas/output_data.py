from pydantic import BaseModel

class PredictionResponse(BaseModel):

    prediction: str
    probability: float
