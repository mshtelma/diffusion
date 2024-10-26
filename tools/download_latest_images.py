
import boto3
import os
import re
from botocore.exceptions import ClientError

def download_s3_files(bucket_name, pattern, local_directory):
    """
    Download files from an S3 bucket that match a specific pattern.

    :param bucket_name: Name of the S3 bucket
    :param pattern: Regex pattern to match file names
    :param local_directory: Local directory to save the files
    """
    s3 = boto3.client('s3')

    # Ensure the local directory exists
    os.makedirs(local_directory, exist_ok=True)

    try:
        # List objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name)

        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if prefix in key and re.search(pattern, key):
                        # Construct the full local file path
                        local_file_path = os.path.join(local_directory, os.path.basename(key))

                        print(f"Downloading {key} to {local_file_path}")

                        # Download the file
                        s3.download_file(bucket_name, key, local_file_path)

        print("Download completed.")

    except ClientError as e:
        print(f"An error occurred: {e}")

# Usage
bucket_name = 'sttk-finetuning'
prefix = "experiments/Nestle_SDXL_Experiment//a9904c1436064299958da3ade7c6e932/artifacts/"
pattern = r'\_100_0.png$'  # This pattern will match all files ending with .csv
local_directory = './downloaded_files'

download_s3_files(bucket_name, pattern, local_directory)