"""
regression.py

This module defines functions to train various regression models using Scikit-Learn.
Supported models include:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor

Evaluation metrics computed include:
    - R² Score
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)

The user may specify a test split size (as a fraction between 0 and 1) via hyperparameters.
If provided, this value will be used to split the dataset into training and testing sets.
If not provided, a default test split size of 0.2 (20%) is used.

Additionally, cross-validation parameters (perform_cv, cv_folds, shuffle) can be provided.
If cross-validation is enabled, the functions compute CV scores and return them.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Import regression models from scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
# Import evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone  # Import clone to create unfitted copies of the model

# --------------------- Utility Function --------------------- #
def _train_and_evaluate(model, X, y, test_size=0.2):
    """
    Splits the dataset using the provided test_size, trains the model, and computes evaluation metrics.

    Parameters:
        model: The scikit-learn regression model instance.
        X: Features (pd.DataFrame or np.array).
        y: Target values (pd.Series or np.array).
        test_size (float): Fraction of data to reserve for testing.

    Returns:
        model: The trained regression model.
        metrics (dict): A dictionary containing:
            - r2_score: R² evaluation metric.
            - mae: Mean Absolute Error.
            - rmse: Root Mean Squared Error.
            - test_size: The number of samples in the test set.
    """
    # Split data with specified test_size and a fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_size": len(X_test)
    }
    return model, metrics

# --------------------- Regression Model Training Functions --------------------- #

def train_linear_regression(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Multiple Linear Regression model.

    Parameters:
        X: Feature data.
        y: Target values.
        user_hyperparams (dict): Hyperparameters provided by the user.
            May include 'fit_intercept' and 'test_size'.
        cross_validation (dict): Cross-validation parameters (perform_cv, cv_folds, shuffle).

    Returns:
        A tuple (model, metrics, cv_results) where cv_results is a dictionary containing CV mean and std, or {} if not performed.
    """
    default_hyperparams = {"fit_intercept": True}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["fit_intercept"] = bool(default_hyperparams.get("fit_intercept", True))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = LinearRegression(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    # Perform cross-validation if enabled, using a clone of the model
    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results

def train_ridge_regression(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Ridge Regression model.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): May include 'alpha', 'solver', 'random_state', 'fit_intercept', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"alpha": 1.0,"max_iter":1000, "solver": "auto", "random_state": 42, "fit_intercept": True}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["max_iter"] = int(default_hyperparams.get("max_iter", 1000))
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    default_hyperparams["fit_intercept"] = bool(default_hyperparams.get("fit_intercept", True))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = Ridge(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results

def train_lasso_regression(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Lasso Regression model.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): May include 'alpha', 'max_iter', 'random_state', 'fit_intercept', and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"alpha": 1.0, "max_iter": 1000, "random_state": 42, "fit_intercept": True}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["max_iter"] = int(default_hyperparams.get("max_iter", 1000))
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    default_hyperparams["fit_intercept"] = bool(default_hyperparams.get("fit_intercept", True))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = Lasso(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results

def train_decision_tree_regressor(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Decision Tree Regressor.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): May include 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state', and 'test_size'.
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
    model = DecisionTreeRegressor(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results

def train_random_forest_regressor(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Random Forest Regressor.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): May include 'n_estimators', 'max_depth', etc., and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["n_estimators"] = int(default_hyperparams.get("n_estimators", 100))
    default_hyperparams["min_samples_split"] = int(default_hyperparams.get("min_samples_split", 2))
    default_hyperparams["min_samples_leaf"] = int(default_hyperparams.get("min_samples_leaf", 1))
    if default_hyperparams.get("max_depth") is not None:
        default_hyperparams["max_depth"] = int(default_hyperparams["max_depth"])
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = RandomForestRegressor(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results

def train_gradient_boosting_regressor(X, y, user_hyperparams, cross_validation={}):
    """
    Trains a Gradient Boosting Regressor.

    Parameters:
        X, y: Training data.
        user_hyperparams (dict): May include 'n_estimators', 'learning_rate', 'max_depth', etc., and 'test_size'.
        cross_validation (dict): Cross-validation parameters.

    Returns:
        A tuple (model, metrics, cv_results).
    """
    default_hyperparams = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
    default_hyperparams.update(user_hyperparams)
    default_hyperparams["n_estimators"] = int(default_hyperparams.get("n_estimators", 100))
    default_hyperparams["max_depth"] = int(default_hyperparams.get("max_depth", 3))
    default_hyperparams["random_state"] = int(default_hyperparams.get("random_state", 42))
    test_size = default_hyperparams.pop("test_size", 0.2)
    model = GradientBoostingRegressor(**default_hyperparams)
    model, metrics = _train_and_evaluate(model, X, y, test_size=test_size)

    cv_results = {}
    if cross_validation.get("perform_cv", False):
        cv_folds = int(cross_validation.get("cv_folds", 5))
        cloned_model = clone(model)
        scores = cross_val_score(cloned_model, X, y, cv=cv_folds, scoring="r2")
        cv_results = {"cv_mean_r2": scores.mean(), "cv_std_r2": scores.std()}

    return model, metrics, cv_results


# --------------------- Main Training Function --------------------- #
def train_regression_model(model_name, X, y, hyperparams={}, cross_validation={}):
    """
    Dynamically selects and trains a regression model.

    Parameters:
        model_name (str): One of "linear_regression", "ridge_regression", "lasso_regression", "decision_tree_regression",
                          "random_forest_regression", or "gradient_boosting_regression".
        X (pd.DataFrame or np.array): Feature data.
        y (pd.Series or np.array): Target data.
        hyperparams (dict): Dictionary of hyperparameters provided by the user. May include "test_size".
        cross_validation (dict): Dictionary containing cross-validation parameters, e.g., {"perform_cv": True, "cv_folds": 5, "shuffle": True}.

    Returns:
        A tuple (model, metrics, cv_results) where:
            model: The trained regression model.
            metrics (dict): Evaluation metrics including r2_score, mae, rmse, and the test set size.
            cv_results (dict): Cross-validation results (mean and standard deviation) if CV is performed; otherwise, {}.
    """
    if model_name == "linear_regression":
        return train_linear_regression(X, y, hyperparams, cross_validation)
    elif model_name == "ridge_regression":
        return train_ridge_regression(X, y, hyperparams, cross_validation)
    elif model_name == "lasso_regression":
        return train_lasso_regression(X, y, hyperparams, cross_validation)
    elif model_name == "decision_tree_regression":
        return train_decision_tree_regressor(X, y, hyperparams, cross_validation)
    elif model_name == "random_forest_regression":
        return train_random_forest_regressor(X, y, hyperparams, cross_validation)
    elif model_name == "gradient_boosting_regression":
        return train_gradient_boosting_regressor(X, y, hyperparams, cross_validation)
    else:
        raise ValueError(f"Unsupported regression model: {model_name}")


# --------------------- Example Usage --------------------- #
if __name__ == "__main__":
    import os

    sample_data_path = r"D:\CODING\Machine Learning\PROJECTS\autoML\data\Salary_Data.csv"
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        user_hyperparams = {"n_estimators": 150, "max_depth": 10, "test_size": 0.25}
        # Example cross-validation parameters: perform CV with 5 folds
        cv_params = {"perform_cv": True, "cv_folds": 5, "shuffle": True}
        model, metrics, cv_results = train_regression_model("random_forest_regression", X, y, user_hyperparams, cv_params)
        print("Random Forest Regression Metrics:", metrics)
        print("Cross-validation Results:", cv_results)
    else:
        print(f"Sample data not found at {sample_data_path}.")
