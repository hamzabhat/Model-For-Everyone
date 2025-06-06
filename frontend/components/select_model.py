import streamlit as st
def render_select_model(api_url, data=None):
    st.markdown('<div class="sub-header">Select and Configure Your Model</div>', unsafe_allow_html=True)

    task_type = st.selectbox(
        "Select Machine Learning Task",
        ["Regression", "Classification", "Clustering", "Association Learning", "Deep Learning"]
    )

    # Define available models based on task type
    model_options = {
        "Regression": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regression",
                       "Random Forest Regression", "Gradient Boosting Regression"],
        "Classification": ["Logistic Regression", "K Nearest Neighbors", "Decision Tree Classifier",
                           "Random Forest Classifier", "Support Vector Machine", "Naive Bayes"],
        "Clustering": ["K Means", "Hierarchical Clustering", "DBSCAN"],
        "Association Learning": ["Apriori","FP-Growth"],
        "Deep Learning": ["Feedforward Neural Network", "Convolutional Neural Network", "Recurrent Neural Network"]
    }

    model_type = st.selectbox("Select Model", model_options[task_type])

    # Get hyperparameters for the selected model
    hyperparams = get_hyperparameters(task_type, model_type)

    # NEW: Let user decide the test split size if it's a supervised task.
    split_size = None
    if task_type in ["Regression", "Classification"]:
        split_size = st.slider("Select Test Split Size (%)", min_value=10, max_value=50, value=20, step=5)
        # Add the split size to hyperparameters to be sent to the backend.
        hyperparams['test_size'] = split_size / 100.0

    # Get cross-validation options if applicable
    if task_type in ["Classification", "Regression"]:
        cv_options = {
            "perform_cv": st.checkbox("Perform Cross-Validation", value=True),
            "cv_folds": st.slider("Number of Folds", 2, 10, 5),
            "shuffle": st.checkbox("Shuffle Data", value=True)
        }
    else:
        cv_options = {"perform_cv": False}

    # Column selection
    feature_columns = []
    target_column = None

    if data is not None:
        data_columns = data.columns.tolist()

        if task_type not in ["Clustering", "Association Learning"]:
            target_column = st.selectbox("Select Target Column", data_columns)
            feature_columns = st.multiselect("Select Feature Columns",
                                             [col for col in data_columns if col != target_column],
                                             default=[col for col in data_columns if col != target_column])
        else:
            feature_columns = st.multiselect("Select Feature Columns", data_columns, default=data_columns)

    # Return the configuration
    config = {
        "task_type": task_type,
        "model_type": model_type,
        "hyperparameters": hyperparams,
        "cross_validation": cv_options,
        "feature_columns": feature_columns,
        "target_column": target_column
    }

    return config

# helper
def get_hyperparameters(task_type, model_type):
    """Get hyperparameters based on the selected model type"""
    hyperparams = {}

    if task_type == "Regression":
        if model_type == "Linear Regression":
            hyperparams['fit_intercept'] = st.checkbox("Fit Intercept", value=True)


        elif model_type in ["Ridge Regression", "Lasso Regression"]:
            hyperparams['alpha'] = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
            hyperparams['fit_intercept'] = st.checkbox("Fit Intercept", value=True)
            hyperparams['max_iter'] = st.slider("Maximum Iterations", 100, 10000, 1000)

        elif model_type in ["Decision Tree Regression", "Random Forest Regression"]:
            hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 30, 5)
            hyperparams['min_samples_split'] = st.slider("Minimum Samples Split", 2, 20, 2)
            hyperparams['min_samples_leaf'] = st.slider("Minimum Samples Leaf", 1, 20, 1)

            if model_type == "Random Forest Regressor":
                hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)

        elif model_type == "Gradient Boosting Regression":
            hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)
            hyperparams['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 30, 3)

    elif task_type == "Classification":
        if model_type == "Logistic Regression":
            hyperparams['C'] = st.slider("C (Inverse of Regularization Strength)", 0.01, 10.0, 1.0)
            hyperparams['max_iter'] = st.slider("Maximum Iterations", 100, 10000, 1000)
            hyperparams['solver'] = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])

        elif model_type == "K Nearest Neighbors":
            hyperparams['n_neighbors'] = st.slider("Number of Neighbors", 1, 20, 5)
            hyperparams['weights'] = st.selectbox("Weight Function", ["uniform", "distance"])
            hyperparams['algorithm'] = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

        elif model_type in ["Decision Tree Classifier", "Random Forest Classifier"]:
            hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 30, 5)
            hyperparams['min_samples_split'] = st.slider("Minimum Samples Split", 2, 20, 2)
            hyperparams['min_samples_leaf'] = st.slider("Minimum Samples Leaf", 1, 20, 1)

            if model_type == "Random Forest Classifier":
                hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)

        elif model_type == "Support Vector Machine":
            hyperparams['C'] = st.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
            hyperparams['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            hyperparams['gamma'] = st.selectbox("Gamma", ["scale", "auto"])

        elif model_type == "Naive Bayes":
            hyperparams['var_smoothing'] = st.slider("Variance Smoothing", 1e-10, 1e-8, 1e-9, format="%.1e")

    elif task_type == "Clustering":
        if model_type == "K-Means":
            hyperparams['n_clusters'] = st.slider("Number of Clusters", 2, 20, 5)
            hyperparams['init'] = st.selectbox("Initialization Method", ["k-means++", "random"])
            hyperparams['max_iter'] = st.slider("Maximum Iterations", 100, 1000, 300)

        elif model_type == "Hierarchical Clustering":
            hyperparams['n_clusters'] = st.slider("Number of Clusters", 2, 20, 5)
            hyperparams['linkage'] = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])

        elif model_type == "DBSCAN":
            hyperparams['eps'] = st.slider("EPS (Maximum Distance Between Points)", 0.1, 2.0, 0.5)
            hyperparams['min_samples'] = st.slider("Minimum Samples", 2, 20, 5)
            hyperparams['algorithm'] = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

    elif task_type == "Association Learning":
        if model_type == "Apriori":
            hyperparams['min_support'] = st.slider("Minimum Support", 0.01, 1.0, 0.1)
            hyperparams['min_confidence'] = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)
            hyperparams['min_lift'] = st.slider("Minimum Lift", 1.0, 10.0, 3.0)

        elif model_type == "FP-Growth":
            hyperparams['min_support'] = st.slider("Minimum Support", 0.01, 1.0, 0.1)
            hyperparams['min_threshold'] = st.slider("Minimum Threshold", 2, 10, 3)

    elif task_type == "Deep Learning":
        hyperparams['epochs'] = st.slider("Number of Epochs", 5, 100, 20)
        hyperparams['batch_size'] = st.slider("Batch Size", 8, 128, 32)
        hyperparams['learning_rate'] = st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        hyperparams['optimizer'] = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])

        if model_type == "Feedforward Neural Network":
            hyperparams['hidden_layers'] = st.slider("Number of Hidden Layers", 1, 5, 2)
            hyperparams['neurons_per_layer'] = st.slider("Neurons Per Layer", 4, 256, 64)
            hyperparams['activation'] = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"])
            hyperparams['dropout_rate'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2)

        elif model_type == "Convolutional Neural Network":
            hyperparams['conv_layers'] = st.slider("Number of Convolutional Layers", 1, 5, 2)
            hyperparams['filters'] = st.slider("Number of Filters", 8, 128, 32)
            hyperparams['kernel_size'] = st.slider("Kernel Size", 2, 7, 3)
            hyperparams['pool_size'] = st.slider("Pooling Size", 2, 4, 2)

        elif model_type == "Recurrent Neural Network":
            hyperparams['rnn_type'] = st.selectbox("RNN Type", ["LSTM", "GRU", "SimpleRNN"])
            hyperparams['rnn_units'] = st.slider("RNN Units", 4, 256, 64)
            hyperparams['rnn_layers'] = st.slider("Number of RNN Layers", 1, 5, 1)
            hyperparams['bidirectional'] = st.checkbox("Bidirectional", value=False)

    return hyperparams