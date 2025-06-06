# backend/api/results.py
"""
This module defines the API endpoint for fetching evaluation metrics and results for a trained model.
It reads a JSON file (created during training) that contains the model's metrics.
"""

from fastapi import APIRouter, HTTPException
import os, json

router = APIRouter()

MODEL_DIR = "backend/models"  # Directory where model metrics are stored


@router.get("/results/{model_id}")
async def get_model_results(model_id: str):
    """
    Results endpoint:
      1. Receives a model_id as a path parameter.
      2. Looks for a corresponding metrics JSON file.
      3. Returns the metrics in the response.
    """
    metrics_file = os.path.join(MODEL_DIR, f"{model_id}_metrics.json")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Model metrics not found.")

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    return {"model_id": model_id, "metrics": metrics}
