import yaml
import os
import sys
import numpy as np
import pickle
import boto3

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# Function to read YAML config files
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Function to write YAML content to file
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        # If replacing is enabled and file exists, remove it
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Save NumPy array to a .npy file
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Save any Python object (e.g. model, transformer) using Pickle
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Load a pickled object from file
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Load NumPy array from .npy file
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Evaluate and compare models using GridSearchCV and R2 Score
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            hyperparams = param.get(model_name, {})

            # Grid search
            gs = GridSearchCV(model, hyperparams, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            # Update model with best params and re-train
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on both train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Train metrics
            train_metrics = {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "precision": precision_score(y_train, y_train_pred),
                "recall": recall_score(y_train, y_train_pred),
                "f1_score": f1_score(y_train, y_train_pred),
            }

            # Test metrics
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred),
                "recall": recall_score(y_test, y_test_pred),
                "f1_score": f1_score(y_test, y_test_pred),
            }

            # Store both train and test metrics
            report[model_name] = {
                "train_score": train_metrics,
                "test_score": test_metrics,
            }

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Generic S3 upload function that works with any bucket type (Gold, Silver, Bronze)
def upload_file_to_s3(file_path: str, s3_key: str, bucket_env_key: str):
    """
    Uploads a file to an S3 bucket.

    Args:
        file_path (str): Path to the local file to be uploaded.
        s3_key (str): Path/key inside the S3 bucket.
        bucket_env_key (str): Environment variable name that holds the S3 bucket name (e.g., 'GOLD_BUCKET').
    """
    try:
        # Load bucket name and credentials from .env or system env
        s3_bucket = os.getenv(bucket_env_key)
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

        # Upload the file to S3
        s3_client.upload_file(file_path, s3_bucket, s3_key)
        logging.info(f" Uploaded {file_path} to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
