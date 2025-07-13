# Import data artifact and config definitions
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig

# Custom exception and logging
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 

# Schema file path
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH

# For statistical comparison between two datasets
from scipy.stats import ks_2samp

# Core libraries
import pandas as pd
import os, sys
from dotenv import load_dotenv
import boto3
from io import StringIO, BytesIO

# Utility functions for YAML handling
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file, upload_file_to_s3

# Load environment variables (.env file)
load_dotenv()


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Constructor that initializes validation configs and reads schema definition.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # Read expected schema from YAML file
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads CSV from the given path and returns it as a DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates if the dataframe has the same number of columns as defined in the schema.
        """
        try:
            number_of_columns = len(self._schema_config)  
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")

            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """
        Uses the Kolmogorov-Smirnov test to compare feature distributions
        between base and current datasets.
        """
        try:
            status = True  # Will be set to False if drift is detected
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                # Perform KS test
                is_same_dist = ks_2samp(d1, d2)

                # If p-value is less than threshold, drift is found
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                # Add drift details for this column
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }

            # Save drift report to file
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Coordinates the full validation process:
        - Load data
        - Check column count
        - Perform drift detection
        - Save validated data
        - Upload to S3
        - Return validation artifact
        """
        try:
            # Paths to raw train/test data
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Load CSVs
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate column count in train
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = "Train dataframe does not contain all columns.\n"

            # Validate column count in test
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = "Test dataframe does not contain all columns.\n"

            # Check for data drift between train and test
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            # Save validated data locally
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # ✅ Upload validated data to Silver S3
            upload_file_to_s3(self.data_validation_config.valid_train_file_path, "validated/train.csv","SILVER_BUCKET")
            upload_file_to_s3(self.data_validation_config.valid_test_file_path, "validated/test.csv", "SILVER_BUCKET")

            # Create and return DataValidationArtifact object
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
