# ModelForge

## Overview

The AutoML Web Application is a **streamlined machine learning platform** that allows users to train, evaluate, and download models without writing code. The application consists of a **FastAPI backend** and a **Streamlit frontend**, providing a complete workflow for dataset processing, model training, and performance evaluation.
## Pending
While the core functionality of ModelForge is up and running, several enhancements and integrations are planned for future releases:
- **Additional Model Types:**  
  - [ ] Add support for additional algorithms in clustering and association rule learning.
- **Deep Learning Models:**  
  - [ ] Add deep learning model support (e.g., CNN, RNN) to the backend training modules.
  
- **Database Integration:**  
  - [ ] Connect and configure the PostgreSQL database for persistent storage of datasets, training jobs, and model metadata.
 
- **User Interface Improvements:**  
  - [ ] Enhance the front-end user experience for better model configuration and training job monitoring.

## Table of Contents
- [Features](#features)
- [Architecture & Project Structure](#architecture--project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
## Features

- **Dataset Upload:**  
  Upload a CSV file through the Streamlit front end. The system displays a preview, data shape, column data types, and missing value statistics.

- **Model Selection & Hyperparameter Tuning:**  
  Choose from various machine learning algorithms for **regression, classification, clustering, and association rule learning**. 

- **Hyperparameter Tuning:**  
  Configure model-specific hyperparameters, test split size, and   cross-validation options via an intuitive user interface.  

- **Asynchronous Training:**  
  Submit training jobs to the backend, which processes them asynchronously. Monitor training progress and view performance metrics upon completion.

- **Cross-Validation:**  
  Optionally enable k-fold cross-validation to evaluate model robustness, with results provided as part of the evaluation metrics.

- **Model Download:**  
  Download the trained model and its evaluation metrics bundled in a single ZIP archive using a unique model identifier.
- **Prediction:**  
  The API also supports model inference via a prediction endpoint (not covered in this README but available in the code).

## Project Structure

```sh
AutoML-Web-App/
├── backend/
│   ├── api/
│   │   ├── upload.py           # API endpoint for uploading datasets
│   │   ├── train.py            # API endpoint for training models asynchronously
│   │   ├── predict.py          # API endpoint for model inference
│   │   ├── download.py         # API endpoint for downloading the trained model and metrics (bundled in a ZIP archive)
│   │   └── results.py          # API endpoint to retrieve training results (if implemented)
│   ├── training/
│   │   ├── regression.py       # Training logic for regression models, including cross-validation support
│   │   ├── classification.py   # Training logic for classification models, including cross-validation support
│   │   ├── clustering.py       # Training logic for clustering models
│   │   └── association.py      # Training logic for association rule learning
│   ├── models/                 # Directory for storing saved models and metrics
│   │   ├── <model_id>.pkl
│   │   └── <model_id>_metrics.json
│   ├── database/               # (Optional) Database connection and ORM models
│   │   ├── db.py
│   │   ├── models.py
│   │   └── create_tables.py
│   └── main.py                 # Entry point for the FastAPI backend
├── frontend/
│   ├── components/
│   │   ├── upload_data.py      # Streamlit component for dataset upload
│   │   ├── select_model.py     # Streamlit component for model selection and hyperparameter configuration
│   │   ├── train_model.py      # Streamlit component to submit training jobs
│   │   └── download_model.py   # Streamlit component to download trained models and metrics
│   └── app.py                  # Main Streamlit front-end application
├── data/
│   └── uploaded_datasets/      # Directory where uploaded CSV datasets are stored
│       ├── dataset_1.csv
│       └── dataset_2.csv
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files and directories to be ignored by Git
```

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/AutoML-Web-App.git
cd AutoML-Web-App
```
### 2. Set Up a Virtual Environment (Recommended)

```sh
python -m venv venv
# Activate the virtual environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Start the Backend Server
Launch the FastAPI backend using Uvicorn:

```sh
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```
The API documentation is available at http://localhost:8080/docs. 
Test the working of the API using **Swagger UI**.

### 5. Start the Frontend Application
Run the Streamlit app:
```
streamlit run frontend/app.py
```
The streamlit application is accessible at http://localhost:8501.

## Usage
### Step 1: Upload Dataset
- **Navigate** to the "Upload Your Dataset" section.
- **Select** a CSV file. The app displays a preview of the dataset along with statistics (e.g., data shape, column data types, and missing values).
- The dataset is **uploaded once** and stored on the backend. A unique `dataset_id` is saved in the session state for subsequent steps.

### Step 2: Select Model & Configure Parameters
- **Choose** the machine learning task (e.g., Regression, Classification).
- **Select** an algorithm from the provided list.
- **Configure hyperparameters** using interactive sliders and checkboxes.
- **Adjust** the test split size and enable cross-validation if desired. These options are added to the configuration payload sent to the backend.

### Step 3: Train Model
- **Click** "Train Model" to submit the training job.
- The backend processes the job **asynchronously** and returns a unique `model_id` (and a `job_id`).
- **Evaluation metrics** and **cross-validation results** are computed during training and saved on the backend.

### Step 4: Download Trained Model
- **Enter** the provided `model_id` in the Download section.
- **Download** a ZIP archive containing both the model file (in pickle format) and the evaluation metrics (in JSON format).

## API Documentation

### Dataset Upload
**Endpoint:** `POST /api/upload`  
**Description:** Uploads a CSV dataset to the backend.  
**Response Example:**

~~~json
{
  "dataset_id": "unique-dataset-uuid",
  "file_path": "data/uploaded_datasets/unique-dataset-uuid.csv"
}
~~~

### Train Model
**Endpoint:** `POST /api/train`  
**Description:** Submits an asynchronous training job with specified configuration.  
**Request Example:**

~~~json
{
  "dataset_id": "unique-dataset-uuid",
  "task_type": "regression",
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 150,
    "max_depth": 10,
    "test_size": 0.2
  },
  "feature_columns": ["Feature1", "Feature2"],
  "target_column": "Target",
  "cross_validation": {
    "perform_cv": true,
    "cv_folds": 5,
    "shuffle": true
  }
}
~~~

**Response Example:**

~~~json
{
  "job_id": "unique-job-uuid",
  "model_id": "unique-model-uuid",
  "message": "Training job submitted successfully."
}
~~~

### Model Prediction
**Endpoint:** `POST /api/predict`  
**Description:** Submits data for inference using a trained model.  
**Response Example:**

~~~json
{
  "predictions": [predicted_value1, predicted_value2]
}
~~~

### Download Model
**Endpoint:** `GET /api/download/{model_id}`  
**Description:** Downloads a ZIP archive containing both the trained model (pickle) and evaluation metrics (JSON).  
**Response:** Returns a ZIP file.

## Contributing
Contributions are welcome! To contribute, follow these steps:  

1. **Fork the repository** on GitHub.  
2. **Clone your fork** locally:  
   ```bash
   git clone https://github.com/hamzabhat/ModelForge.git
   ```

3. **Create a new branch** for your feature or fix:
```
git checkout -b feature-branch
```
4. **Make changes and commit them**:
```
git commit -m "Added a new feature"
```
5. **Push to your fork** and **create a pull request**:
```
git push origin feature-branch
```
4. **Submit a pull request** and describe your changes.

#### Feel free to open an issue for any improvements or questions!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for further details