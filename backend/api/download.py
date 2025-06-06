# backend/api/download.py
"""
This module defines the API endpoint for downloading a trained model.
It now returns a ZIP archive containing both the model file (pickle)
and the evaluation metrics (JSON), so the user doesn't have to choose a format.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import os, json, io, zipfile

router = APIRouter()

MODEL_DIR = "backend/models"  # Directory where models and metrics are saved


@router.get("/download/{model_id}")
async def download_model(model_id: str):
    """
    Download endpoint:
      1. Accepts a model_id.
      2. Checks for the existence of the model file (<model_id>.pkl) and the metrics file (<model_id>_metrics.json).
      3. Bundles both files into an in-memory ZIP archive.
      4. Returns the ZIP archive as a downloadable file.

    This implementation ensures that both the trained model and its evaluation metrics
    are provided to the user in a single download.
    """
    model_id=model_id.strip()
    # Build file paths for the model and metrics files
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"{model_id}_metrics.json")

    # Check if both files exist; if not, raise a 404 error
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found.")

    # Create an in-memory ZIP archive containing both files
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add the model file with an archive name (without full path)
        zip_file.write(model_path, arcname=f"{model_id}.pkl")
        # Add the metrics file with an archive name
        zip_file.write(metrics_path, arcname=f"{model_id}_metrics.json")

    zip_buffer.seek(0)  # Reset buffer pointer to the beginning

    # Return the ZIP archive as a downloadable streaming response.
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={model_id}.zip"}
    )
