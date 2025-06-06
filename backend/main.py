# backend/main.py
"""
Main entry point for the FastAPI application.
This file mounts the modular API routers from the backend/api/ directory.
It creates the FastAPI app instance and includes routers for training, prediction,
model download, and retrieving evaluation results.
"""
from fastapi import FastAPI
# Import API routers from the backend/api directory.
# Ensure that __init__.py is present in the backend/api/ folder to treat it as a module.
from backend.api import upload, train, predict, download, results

# Create the FastAPI app instance with metadata.
app = FastAPI(
    title="ML Web App API",
    description="API for training, prediction, and model management for the ML Web App",
    version="1.0.0"
)

# Mount each API router under a common prefix (e.g., /api)
app.include_router(upload.router, prefix="/api")
app.include_router(train.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(download.router, prefix="/api")
app.include_router(results.router, prefix="/api")

# Define a simple root endpoint.
@app.get("/")
async def root():
    return {"message": "Welcome to the ML Web App API"}
