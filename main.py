from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List



model = joblib.load('rf_model.pkl')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def helloWorld():
    return {"message": "Simple API for Credit Fraud Detection"}

@app.post("/predict") 
def predict(request: PredictRequest):
    data = np.array([request.features])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}
