"""
models.py

This script defines ORM models (tables) for storing dataset information and model configuration.
We define two models:
  - Dataset: Stores information about uploaded datasets.
  - ModelConfig: Stores configuration details and performance metrics for trained models.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from backend.database.db import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String, unique=True, index=True, nullable=False)  # Unique identifier from API (UUID)
    filename = Column(String, nullable=False)
    file_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Dataset(id={self.id}, uuid={self.dataset_uuid}, filename='{self.filename}')>"

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True)
    model_uuid = Column(String, unique=True, index=True, nullable=False)  # Unique identifier for the trained model
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    task_type = Column(String, nullable=False)       # e.g., regression, classification
    model_type = Column(String, nullable=False)        # e.g., random_forest, linear_regression, etc.
    hyperparameters = Column(JSON, nullable=False)     # Store hyperparameters as JSON
    metrics = Column(JSON, nullable=True)              # Evaluation metrics after training (e.g., R² score)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelConfig(id={self.id}, model_uuid='{self.model_uuid}', model_type='{self.model_type}')>"
