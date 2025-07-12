# Custom exceptions and logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Data ingestion configuration and artifact tracking
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

# Core imports
import os
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import boto3
from io import StringIO, BytesIO

# Load environment variables from .env
load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize with configuration details for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def load_data_from_s3(self) -> pd.DataFrame:
        """
        Loads raw data from an S3 bucket.
        Supports .csv and .json file formats.
        Returns:
            Pandas DataFrame of the loaded data.
        """
        try:
            # Read bucket and key from environment
            s3_bucket = os.getenv("S3_BUCKET")
            s3_key = os.getenv("S3_KEY")

            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )

            # Read file content from S3
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            file_content = obj['Body'].read().decode('utf-8')

            # Load content into DataFrame
            if s3_key.endswith('.json'):
                df = pd.read_json(StringIO(file_content))
            elif s3_key.endswith('.csv'):
                df = pd.read_csv(StringIO(file_content))
            else:
                raise ValueError("Unsupported file format. Must be .csv or .json")

            # Replace 'na' string with np.nan
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Saves the cleaned/loaded data locally into the feature store (Silver layer).
        Returns:
            The same DataFrame for further processing.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def upload_file_to_s3(self, file_path: str, s3_key: str):
        """
        Uploads a local file to the specified key in the Bronze S3 bucket.
        Args:
            file_path: Local file path to upload.
            s3_key: Destination path inside the S3 bucket.
        """
        try:
            s3_bucket = os.getenv("BRONZE_BUCKET")
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )

            s3_client.upload_file(file_path, s3_bucket, s3_key)
            logging.info(f"Uploaded {file_path} to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the dataset into training and testing sets, saves them locally,
        and uploads them to the Bronze S3 bucket.
        """
        try:
            # Split the dataset
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            # Save to local paths
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            # Upload train/test splits to S3 (Bronze layer)
            self.upload_file_to_s3(train_path, "train_test/train.csv")
            self.upload_file_to_s3(test_path, "train_test/test.csv")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        """
        Full pipeline: load → transform → save → split → upload.
        Returns:
            DataIngestionArtifact containing paths to training and testing datasets.
        """
        try:
            dataframe = self.load_data_from_s3()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
