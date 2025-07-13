# Required standard libraries
import sys
import os
import numpy as np
import pandas as pd

# Sklearn for missing value imputation and pipelines
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# Load environment variables (for AWS S3 credentials)
from dotenv import load_dotenv
import boto3

# Project-specific constants
from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

# Artifact and config entities
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataTransformationConfig

# Custom exception and logger
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

# Utility functions to save objects and arrays
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

# Load environment variables from `.env` file
load_dotenv()

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Constructor that initializes with data validation artifacts (paths to valid train/test files)
        and transformation configuration (where to save outputs).
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads a CSV file and returns it as a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initializes a pipeline with KNNImputer based on config parameters.
        Used to handle missing values.
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def upload_file_to_s3(self, file_path: str, s3_key: str):
        """
        Uploads a local file (npy or pkl) to a specific key in the AWS S3 bucket.
        """
        try:
            s3_bucket = os.getenv("GOLD_BUCKET")  # Bucket for transformed (gold) data
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )

            s3_client.upload_file(file_path, s3_bucket, s3_key)
            logging.info(f"✅ Uploaded {file_path} to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method that:
        - Loads validated data
        - Splits into features and target
        - Applies missing value imputation
        - Saves transformed data and pipeline object
        - Uploads results to S3
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")

            # Read validated train/test CSV files
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split train set: features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)  # normalize -1 to 0

            # Split test set: features and target
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Create pipeline with KNNImputer
            preprocessor = self.get_data_transformer_object()

            # Fit on train data and transform both train/test
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Combine features and targets into single arrays
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed data locally as .npy files
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save preprocessor pipeline object (.pkl)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Upload transformed files to S3
            self.upload_file_to_s3(self.data_transformation_config.transformed_train_file_path, "transformed/train.npy")
            self.upload_file_to_s3(self.data_transformation_config.transformed_test_file_path, "transformed/test.npy")
            self.upload_file_to_s3(self.data_transformation_config.transformed_object_file_path, "transformed/preprocessor.pkl")

            # Prepare artifact object to pass to the next pipeline stage
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
