# frontend/app.py
import streamlit as st
import requests
from components.upload_data import render_upload_data
from components.select_model import render_select_model
from components.train_model import render_train_model
from components.download_model import render_download_model

api_url = "http://localhost:8080/api"

st.title("ML Web App")

# Step 1: Upload Dataset
st.header("Step 1: Upload Your Dataset")
if render_upload_data(api_url):
    st.success("Dataset uploaded. You can now configure your model.")

    # Debug: Show session state after upload
    st.write("Session state after upload:", st.session_state)

    # Retrieve dataset and proceed only if present
    data = st.session_state.get("uploaded_data")

    # Step 2: Select and Configure Model
    st.header("Step 2: Select and Configure Your Model")
    config = render_select_model(api_url, data)
    st.write("Your model configuration:")
    st.json(config)

    # Step 3: Submit Training Job
    st.header("Step 3: Train Your Model")
    dataset_id = st.session_state.get("dataset_id")
    if dataset_id:
        job_id = render_train_model(api_url, config, dataset_id)
        if job_id:
            st.info(f"Training job submitted successfully! Job ID: {job_id}")
    else:
        st.error("Dataset ID not found. Please upload your dataset again.")

    # Step 4: Download Trained Model
    st.header("Step 4: Download Your Trained Model")
    model_id = st.text_input("Enter the Model ID to download:", value="")
    if model_id:
        render_download_model(api_url, model_id)
else:
    st.warning("Please upload a dataset to begin.")
