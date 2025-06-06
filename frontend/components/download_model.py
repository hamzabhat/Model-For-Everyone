# components/download_model.py
"""
This module defines the front-end component for downloading a trained model.
Instead of asking the user for a download format, it calls the backend download endpoint,
which returns a ZIP archive that bundles both the model file (pickle) and the metrics file (JSON).
The user only needs to enter the Model ID.
"""

import streamlit as st
import requests
import base64
import io
import zipfile


def render_download_model(api_url, model_id):
    st.markdown('<div class="sub-header">Download Your Trained Model</div>', unsafe_allow_html=True)

    try:
        # Call the backend endpoint that creates a ZIP archive containing both files.
        # In our updated backend, the endpoint /download/{model_id} will now return the ZIP archive.
        response = requests.get(f"{api_url}/download/{model_id}")

        if response.status_code == 200:
            # Encode the binary ZIP data to base64 for display in a download link
            b64 = base64.b64encode(response.content).decode()
            # Create a download link using an HTML anchor tag with a data URL
            href = f'<a href="data:application/zip;base64,{b64}" download="{model_id}.zip">Click here to download your model and metrics (ZIP)</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Optionally, you can display instructions on how to extract and use the files
            st.markdown('<div class="success-box">', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Error downloading model: {response.text}")

    except Exception as e:
        st.error(f"Error fetching model: {str(e)}")
