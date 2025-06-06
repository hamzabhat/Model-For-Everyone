"""
create_tables.py

Script to create all tables in the database using SQLAlchemy ORM.
Run this script once to initialize your database schema.
"""

from backend.database.db import engine, Base
from backend.database import models  # Ensure models are imported so that Base knows about them

# Create all tables
Base.metadata.create_all(bind=engine)
print("Tables created successfully.")
