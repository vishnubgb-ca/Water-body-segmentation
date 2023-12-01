import boto3
import os

os.environ["AWS_ACCESS_KEY_ID"]="AKIA3YG72WSKJ2FNO6US"
os.environ["AWS_SECRET_ACCESS_KEY"]="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"

import boto3

def extract_data():

    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops'
    url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': 'Water_Bodies_Dataset_Split.zip'},
                    ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
                )
    print(url)
    return url

extract_data()
