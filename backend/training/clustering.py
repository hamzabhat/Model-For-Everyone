"""
clustering.py

This module defines functions to train various clustering models using Scikit-Learn.
Supported models:
    - K-Means
    - Hierarchical Clustering (Agglomerative)
    - DBSCAN

Since clustering is unsupervised, the output typically includes cluster labels.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def train_kmeans(X, user_hyperparams):
    """Train K-Means clustering model."""
    default_hyperparams = {
        "n_clusters": 5,
        "init": "k-means++",
        "max_iter": 300,
        "random_state": 42
    }
    default_hyperparams.update(user_hyperparams)
    model = KMeans(**default_hyperparams)
    model.fit(X)
    labels = model.labels_
    metrics = {"inertia": model.inertia_, "n_clusters": default_hyperparams["n_clusters"]}
    return model, {"labels": labels, **metrics}


def train_hierarchical(X, user_hyperparams):
    """Train Hierarchical Clustering model (Agglomerative)."""
    default_hyperparams = {
        "n_clusters": 5,
        "linkage": "ward"
    }
    default_hyperparams.update(user_hyperparams)
    model = AgglomerativeClustering(**default_hyperparams)
    labels = model.fit_predict(X)
    metrics = {"n_clusters": default_hyperparams["n_clusters"]}
    return model, {"labels": labels, **metrics}


def train_dbscan(X, user_hyperparams):
    """Train DBSCAN clustering model."""
    default_hyperparams = {
        "eps": 0.5,
        "min_samples": 5,
        "algorithm": "auto"
    }
    default_hyperparams.update(user_hyperparams)
    model = DBSCAN(**default_hyperparams)
    labels = model.fit_predict(X)
    metrics = {"n_clusters": len(set(labels)) - (1 if -1 in labels else 0)}
    return model, {"labels": labels, **metrics}


def train_clustering_model(model_name, X, hyperparams={}):
    """
    Dynamically selects and trains a clustering model.

    model_name (str): Options include "kmeans", "hierarchical", or "dbscan".
    X: Data to cluster.
    hyperparams (dict): Hyperparameters from the front end.

    Returns:
      model: The clustering model.
      metrics (dict): Metrics including cluster labels and model-specific details.
    """
    if model_name == "k_means":
        return train_kmeans(X, hyperparams)
    elif model_name == "hierarchical_clustering":
        return train_hierarchical(X, hyperparams)
    elif model_name == "dbscan":
        return train_dbscan(X, hyperparams)
    else:
        raise ValueError(f"Unsupported clustering model: {model_name}")


# --------------------- Example Usage --------------------- #
if __name__ == "__main__":
    import os

    sample_data_path = os.path.join("data", "dataset_clustering.csv")
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        # Assume all columns are features for clustering
        X = df.values
        user_hyperparams = {"n_clusters": 4}
        model, metrics = train_clustering_model("kmeans", X, user_hyperparams)
        print("K-Means Clustering Metrics:", metrics)
    else:
        print(f"Sample data not found at {sample_data_path}.")
