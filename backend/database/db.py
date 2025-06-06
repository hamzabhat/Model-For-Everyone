"""
db.py

This script sets up the connection to a PostgreSQL database using SQLAlchemy.
It creates an engine, a session factory, and a declarative base that will be used for ORM models.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database connection details from environment variables (or set defaults for local development)
DATABASE_USER = os.getenv("DATABASE_USER", "your_username")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "your_password")
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ml_web_app_db")

# Create the database URL for PostgreSQL
SQLALCHEMY_DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# Create the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our ORM models
Base = declarative_base()

# Dependency function to get a session (useful in FastAPI endpoints)
def get_db():
    """
    Provides a database session.
    Ensures that the session is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
