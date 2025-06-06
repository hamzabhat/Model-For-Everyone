import streamlit as st
import requests

def render_train_model(api_url, config, dataset_id):
    st.markdown('<div class="sub-header">Train Your Model</div>', unsafe_allow_html=True)

    if not config["feature_columns"]:
        st.error("Please select at least one feature column.")
        return None

    if st.button("Train Model"):
        # Create training job payload
        training_payload = {
            "dataset_id": dataset_id,
            "task_type": config["task_type"].lower(),
            "model_type": config["model_type"].lower().replace(" ", "_"),
            "hyperparameters": config["hyperparameters"],
            "feature_columns": config["feature_columns"],
            "target_column": config["target_column"],
            "cross_validation": config["cross_validation"]
        }

        # Send training request to API to get JSONRESPONSE
        with st.spinner("Submitting training job..."):
            try:
                response = requests.post(f"{api_url}/train", json=training_payload)

                if response.status_code == 200:
                    job_data = response.json()

                    # Store both job_id and model_id (if present)
                    job_id = job_data.get("job_id")
                    model_id = job_data.get("model_id")

                    if job_id:
                        st.session_state.training_job_id = job_id
                        st.success(f"Training complete! Job ID: {job_id}")

                    if model_id:
                        st.session_state.trained_model_id = model_id
                        st.success(f"Model saved! Model ID: {model_id}")

                    if not job_id and not model_id:
                        st.error("Error: No job ID or model ID received from backend.")

                    # Display a note for model download
                    if model_id:
                        st.write("Copy this Model ID for downloading your trained model in Step 4.")

                else:
                    st.error(f"Error submitting training job: {response.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
