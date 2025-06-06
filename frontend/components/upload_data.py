import streamlit as st
import pandas as pd
import requests

def render_upload_data(api_url):
    st.markdown('<div class="sub-header">Upload Your Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read and display the data
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            st.write("Data Preview:")
            st.dataframe(data.head())

            # Data statistics
            st.write("Data Shape:", data.shape)

            # Check if the dataset is already uploaded
            if "dataset_id" not in st.session_state:
                file_content = uploaded_file.getvalue()
                files = {'file': ('dataset.csv', file_content, 'text/csv')}
                response = requests.post(f"{api_url}/upload", files=files)

                if response.status_code == 200:
                    dataset_id = response.json()['dataset_id']
                    st.session_state.dataset_id = dataset_id
                    st.success(f"Dataset uploaded successfully! ID: {dataset_id}")
                else:
                    st.error(f"Error uploading dataset: {response.text}")
            else:
                st.info(f"Using previously uploaded dataset: {st.session_state.dataset_id}")

            # Display dataset information
            col1, col2 = st.columns(2)

            with col1:
                st.write("Column Data Types:")
                dtypes_df = pd.DataFrame(data.dtypes, columns=["Data Type"])
                st.dataframe(dtypes_df)

            with col2:
                st.write("Missing Values:")
                missing_df = pd.DataFrame(data.isnull().sum(), columns=["Count"])
                missing_df['Percentage'] = (missing_df['Count'] / len(data) * 100).round(2)
                st.dataframe(missing_df)

        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

    return uploaded_file is not None or "dataset_id" in st.session_state
    # Return True if dataset exists
