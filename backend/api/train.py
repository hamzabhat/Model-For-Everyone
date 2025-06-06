"""
train.py

This module defines the API endpoint for training machine learning models.
It accepts a JSON payload from the front end, which includes:
    - dataset_id: Identifier of the uploaded dataset (stored as a CSV in `data/`)
    - task_type: Machine learning task ("regression", "classification", etc.)
    - model_type: Model name (e.g., "random_forest", "logistic_regression", etc.)
    - hyperparameters: Dictionary of user-defined hyperparameters
    - feature_columns: Selected features for training
    - target_column: Target column (only for supervised learning)
    - cross_validation: Whether to use cross-validation (and associated parameters)

Once validated, it loads the dataset, extracts features (and target if needed),
trains the model in a background task, and saves both the trained model and evaluation metrics.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import os, json, uuid
import pandas as pd
import pickle

# Import training functions for different ML tasks
from backend.training.regression import train_regression_model
from backend.training.classification import train_classification_model
from backend.training.clustering import train_clustering_model
from backend.training.association import train_association_model

router = APIRouter()

# Define directories for storing datasets and models
DATASET_DIR = "data"
MODEL_DIR = "backend/models"


# Updated Pydantic Model for API Request Validation to include cross_validation.
class TrainRequest(BaseModel):
    dataset_id: str
    task_type: str
    model_type: str
    hyperparameters: Dict[str, float]
    feature_columns: List[str]
    target_column: Optional[str] = None  # Optional for unsupervised tasks
    cross_validation: Dict[str, float] = {"perform_cv": 0}  # Use 0 for False, or you can later convert to bool in code


@router.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    API endpoint for training a model.

    Steps:
    1. Parses JSON payload from the request.
    2. Validates required fields.
    3. Loads dataset using the dataset_id.
    4. Extracts feature columns & target column (if applicable).
    5. Reads cross-validation parameters from payload.
    6. Starts a background task for model training.
    7. Saves trained model and evaluation metrics (including cross-validation results).
    8. Returns `model_id` and `job_id` to the frontend.
    """
    try:
        # Parse JSON payload for train_model and convert to dictionary
        payload = request.dict()

        # Validate required fields
        required_fields = ["dataset_id", "task_type", "model_type", "hyperparameters", "feature_columns"]
        for field in required_fields:
            if field not in payload:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Ensure target_column is provided for supervised tasks
        task_type = payload["task_type"].lower()
        if task_type in ["regression", "classification"]:
            if "target_column" not in payload or not payload["target_column"]:
                raise HTTPException(status_code=400, detail="Missing target_column for supervised learning task")

        # Construct dataset path
        dataset_file = os.path.join(DATASET_DIR, f"{payload['dataset_id']}.csv")
        if not os.path.exists(dataset_file):
            raise HTTPException(status_code=404, detail="Dataset not found. Please upload the dataset first.")

        # Load dataset using pandas
        try:
            data = pd.read_csv(dataset_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading dataset: {e}")

        # Validate feature columns
        feature_columns = payload["feature_columns"]
        if not all(col in data.columns for col in feature_columns):
            raise HTTPException(status_code=400, detail="One or more feature columns not found in dataset")
        X = data[feature_columns]

        # Extract target column if applicable
        if task_type in ["regression", "classification"]:
            target_column = payload["target_column"]
            if target_column not in data.columns:
                raise HTTPException(status_code=400, detail="Target column not found in dataset")
            y = data[target_column]
        else:
            y = None

        # Read cross-validation parameters from payload (if provided)
        # Expected keys: perform_cv, cv_folds, shuffle
        # Convert perform_cv to bool if needed
        cross_validation = payload.get("cross_validation", {"perform_cv": 0})
        cross_validation["perform_cv"] = bool(cross_validation.get("perform_cv", 0))

        # Generate model ID
        model_id = str(uuid.uuid4())

        def background_train():
            """
            Background task to train the model based on task type and model type.
            Saves the trained model and metrics.
            """
            try:
                # Normalize model_type string (e.g., "Random Forest" -> "random_forest")
                model_type = payload["model_type"].lower().replace(" ", "_")
                hyperparameters = payload.get("hyperparameters", {})

                # Select appropriate training function based on task_type
                if task_type == "regression":
                    # Pass cross_validation parameters to the regression training function
                    trained_model, metrics, cv_results = train_regression_model(
                        model_type, X, y, hyperparameters, cross_validation
                    )
                elif task_type == "classification":
                    trained_model, metrics, cv_results = train_classification_model(
                        model_type, X, y, hyperparameters, cross_validation
                    )
                elif task_type == "clustering":
                    trained_model, metrics, cv_results = train_clustering_model(model_type, X, hyperparameters)
                elif task_type == "association":
                    transactions = X.astype(str).values.tolist()
                    trained_model, metrics, cv_results = train_association_model(model_type, transactions,
                                                                                 hyperparameters)
                else:
                    print("❌ Unsupported task type")
                    return

                # Save the trained model using pickle
                model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(trained_model, f)

                # Save evaluation metrics along with cross-validation results
                metrics_file = os.path.join(MODEL_DIR, f"{model_id}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump({"evaluation_metrics": metrics, "cross_validation_results": cv_results}, f)

                print(f"✅ Training complete. Model saved with ID: {model_id}")
            except Exception as e:
                print(f"❌ Error during training: {e}")

        # Start background training task
        background_tasks.add_task(background_train)

        # Generate a job ID to track the training request
        job_id = str(uuid.uuid4())

        # Return response with job_id and model_id
        return JSONResponse(
            content={"job_id": job_id, "model_id": model_id, "message": "Training job submitted successfully."}
        )


    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {e}"})
