"""
classification.py

This module defines functions to train various classification models using Scikit-Learn.
Supported models include:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Machine (SVM)
    - Naive Bayes

Evaluation metrics computed include:
    - Accuracy
    - Precision
    - Recall
    - F1-Score

This script accepts a dynamic test split size from the front end.

Additionally, it accepts cross-validation parameters.
If cross-validation is enabled, the functions compute cross-validation scores and return them.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

# Import classification models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import evaluation metrics for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------- Utility Function --------------------- #
def _train_and_evaluate(model, X, y, test_size=0.2):
    """
    Splits the data using the provided test_size, trains the classifier, and computes evaluation metrics.

    Parameters:
        model: The scikit-learn classifier instance.
        X: Feature data.
        y: Target labels.
        test_size (float): Fraction of the dataset to reserve for testing.

    Returns:
        model: The trained classifier.
        metrics (dict): Evaluation metrics including accuracy, precision, recall, f1_score, and the number of test samples.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "test_size": len(X_test)
    }
    return model, metrics


# --------------------- Classification Model Training Functions --------------------- #

def train_logistic_regression(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Logistic Regression classifier.

    Parameters:
        X: Feature data.
        y: Target labels.
        user_hyperparams (dict): Hyperparameters, may include 'C', 'max_iter', 'solver', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["max_iter"] = int(default_hyperparams.get("max_iter", 1000))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = LogisticRegression(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


def train_knn(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a K-Nearest Neighbors classifier.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): Hyperparameters, may include 'n_neighbors', 'weights', 'algorithm', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["n_neighbors"] = int(default_hyperparams.get("n_neighbors", 5))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = KNeighborsClassifier(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


def train_decision_tree_classifier(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Decision Tree Classifier.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): Hyperparameters, may include 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["min_samples_split"] = int(default_hyperparams.get("min_samples_split", 2))
    default_hyperparams["min_samples_leaf"] = int(default_hyperparams.get("min_samples_leaf", 1))
    if default_hyperparams.get("max_depth") is not None:
        default_hyperparams["max_depth"] = int(default_hyperparams["max_depth"])
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = DecisionTreeClassifier(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


def train_random_forest_classifier(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Random Forest Classifier.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): Hyperparameters, may include 'n_estimators', 'max_depth', etc., and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,
                           "random_state": 42}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["n_estimators"] = int(default_hyperparams.get("n_estimators", 100))
    default_hyperparams["min_samples_split"] = int(default_hyperparams.get("min_samples_split", 2))
    default_hyperparams["min_samples_leaf"] = int(default_hyperparams.get("min_samples_leaf", 1))
    if default_hyperparams.get("max_depth") is not None:
        default_hyperparams["max_depth"] = int(default_hyperparams["max_depth"])
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = RandomForestClassifier(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


def train_svm(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Support Vector Machine (SVM) classifier.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): Hyperparameters, may include 'C', 'kernel', 'gamma', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["C"] = float(default_hyperparams.get("C", 1.0))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = SVC(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


def train_naive_bayes(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Gaussian Naive Bayes classifier.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): Hyperparameters, may include 'var_smoothing' and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"var_smoothing": 1e-9}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["var_smoothing"] = float(default_hyperparams.get("var_smoothing", 1e-9))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = GaussianNB(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        cv_results = {"cv_mean_accuracy": scores.mean(), "cv_std_accuracy": scores.std()}

    return model, metrics, cv_results


# --------------------- Main Training Function --------------------- #
def train_classification_model(model_name, X, y, hyperparams={}, cross_validation={}):
    """
    Dynamically selects and trains a classification model.

    Parameters:
        model_name (str): One of "logistic_regression", "knn", "decision_tree_classifier", "random_forest_classifier", "support_vector_machine", or "naive_bayes".
        X (pd.DataFrame or np.array): Feature data.
        y (pd.Series or np.array): Target labels.
        hyperparams (dict): Dictionary of hyperparameters provided by the user, may include "test_size".
        cross_validation (dict): Dictionary containing cross-validation parameters, e.g., {"perform_cv": True, "cv_folds": 5, "shuffle": True}.

    Returns:
        A tuple (model, metrics, cv_results) where:
            model: The trained classifier.
            metrics (dict): Evaluation metrics including accuracy, precision, recall, f1_score, and the test set size.
            cv_results (dict): Cross-validation results (mean and standard deviation) if CV is performed; otherwise, {}.
    """
    if model_name == "logistic_regression":
        return train_logistic_regression(X, y, hyperparams, cross_validation)
    elif model_name == "k_nearest_neighbors":
        return train_knn(X, y, hyperparams, cross_validation)
    elif model_name == "decision_tree_classifier":
        return train_decision_tree_classifier(X, y, hyperparams, cross_validation)
    elif model_name == "random_forest_classifier":
        return train_random_forest_classifier(X, y, hyperparams, cross_validation)
    elif model_name == "support_vector_machine":
        return train_svm(X, y, hyperparams, cross_validation)
    elif model_name == "naive_bayes":
        return train_naive_bayes(X, y, hyperparams, cross_validation)
    else:
        raise ValueError(f"Unsupported classification model: {model_name}")


# --------------------- Example Usage --------------------- #
if __name__ == "__main__":
    import os

    sample_data_path = os.path.join("data", "dataset_classification.csv")
    if os.path.exists(sample_data_path):
        # Load sample classification dataset
        df = pd.read_csv(sample_data_path)
        # Assume the last column is the target and the rest are features
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        # Example user-specified hyperparameters including a 30% test split
        user_hyperparams = {"C": 0.5, "max_iter": 500, "test_size": 0.3}
        # Example cross-validation parameters: perform CV with 5 folds
        cv_params = {"perform_cv": True, "cv_folds": 5, "shuffle": True}
        model, metrics, cv_results = train_classification_model("logistic_regression", X, y, user_hyperparams,
                                                                cv_params)
        print("Logistic Regression Metrics:", metrics)
        print("Cross-validation Results:", cv_results)
    else:
        print(f"Sample data not found at {sample_data_path}.")
