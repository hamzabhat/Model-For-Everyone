# backend/api/predict.py
"""
This module defines the API endpoint for making predictions with a trained model.
It expects a JSON payload that includes:
    - model_id: Identifier of the saved model file.
    - input_data: Data for which predictions are needed (in list-of-dict or table format).
The endpoint loads the model and returns the predictions.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os, pickle
import pandas as pd

router = APIRouter()

MODEL_DIR = "backend/models"  # Directory where trained models are stored

# Define a Pydantic model for the predict request
class PredictRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Any]]  # A list of dictionaries representing rows

@router.post("/predict")
async def predict_model(request: PredictRequest):
    """
    Predict endpoint:
      1. Receives a JSON payload with model_id and input_data.
      2. Loads the trained model from disk.
      3. Converts input_data into a DataFrame.
      4. Calls model.predict() and returns the predictions.
    """
    payload = request.dict()

    # Validate required fields (Pydantic does that, but extra check here if needed)
    if not payload.get("model_id") or not payload.get("input_data"):
        raise HTTPException(status_code=400, detail="Missing model_id or input_data.")

    model_id = payload["model_id"]
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")

    # Load the model using pickle
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Convert input_data to a DataFrame
    try:
        input_data = pd.DataFrame(payload["input_data"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")

    # Make predictions using the model
    try:
        predictions = model.predict(input_data).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    return JSONResponse(content={"predictions": predictions})
