import boto3
import os



import boto3

def extract_data():

    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops-demo'
    url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': 'Water_Bodies_Dataset_Split.zip'},
                    ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
                )
    print(url)
    return url

extract_data()
