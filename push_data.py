import os
import sys
import json
import uuid
import boto3

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

import pandas as pd
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkDataExtract():
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def upload_data_to_s3(self, records, s3_key_prefix="network-data"):
        try:
            json_data = json.dumps(records, indent=2)
            unique_file_name = f"{s3_key_prefix}/records_{uuid.uuid4().hex}.json"

            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=unique_file_name,
                Body=json_data,
                ContentType='application/json'
            )
            logging.info(f"Uploaded {len(records)} records to S3 at {unique_file_name}")
            return unique_file_name
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    networkobj = NetworkDataExtract()

    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)

    s3_key = networkobj.upload_data_to_s3(records)
    print(f"Uploaded to: s3://{S3_BUCKET}/{s3_key}")
