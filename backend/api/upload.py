# backend/api/upload.py
"""
This module defines the API endpoint for dataset upload.
It receives a CSV file (as multipart/form-data) from the front end,
saves the file locally in the "data/" directory, and returns a unique dataset_id.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
import os, shutil, uuid

# Create an API router for upload endpoints.
router = APIRouter()

# Directory where uploaded datasets will be stored.
DATASET_DIR = "data"
# Ensure the DATASET_DIR exists.
os.makedirs(DATASET_DIR, exist_ok=True)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Endpoint: /upload
    Method: POST

    This endpoint performs the following:
      1. Receives a CSV file from the front end.
      2. Generates a unique dataset_id using UUID.
      3. Saves the file in the DATASET_DIR with a name based on the dataset_id.
      4. Returns a JSON response with the dataset_id (and optionally file path).

    Parameters:
        file (UploadFile): The uploaded CSV file.

    Returns:
        JSON: {"dataset_id": <unique_id>, "file_path": <location on disk>}

    Raises:
        HTTPException: If there is an error while saving the file.
    """
    # Generate a unique identifier for the dataset.
    dataset_id = str(uuid.uuid4())
    # Construct the file location: e.g., data/<dataset_id>.csv
    file_location = os.path.join(DATASET_DIR, f"{dataset_id}.csv")

    try:
        # Open the file for writing in binary mode and copy the contents.
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Return the unique dataset_id so that the front end can reference this dataset later.
    return {"dataset_id": dataset_id, "file_path": file_location}
