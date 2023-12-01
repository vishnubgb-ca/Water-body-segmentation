import boto3
import os

os.environ["AWS_ACCESS_KEY_ID"]="AKIA3YG72WSKJ2FNO6US"
os.environ["AWS_SECRET_ACCESS_KEY"]="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"

import boto3

def extract_data():

    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops'

    # glioma_tumor_folder_prefix_training = 'Brain tumor classification/Training/glioma_tumor/'  
    # meningioma_tumor_folder_prefix_training = 'Brain tumor classification/Training/meningioma_tumor/'
    # no_tumor_folder_prefix_training = 'Brain tumor classification/Training/no_tumor/'
    # pituitory_tumor_folder_prefix_training = 'Brain tumor classification/Training/pituitary_tumor/'

    def get_object_presigned_url(bucket_name,folder_prefix):

        # List objects within the folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

        # Generate presigned URLs for each object in the folder
        presigned_urls = []
        for obj in response.get('Contents', []):
            # Ensure that the key is not the folder itself but its contents
            if obj['Key'] != folder_prefix:
                # Generate presigned URL for the object
                url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': obj['Key']},
                    ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
                )
                presigned_urls.append(url)
        return presigned_urls
    url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': 'Training.zip'},
                    ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
                )
    # glioma_tumor = get_object_presigned_url(bucket_name,glioma_tumor_folder_prefix_training)
    # glioma_tumor = get_object_presigned_url(bucket_name,glioma_tumor_folder_prefix_training)
    # meningioma_tumor = get_object_presigned_url(bucket_name,meningioma_tumor_folder_prefix_training)
    # no_tumor = get_object_presigned_url(bucket_name,no_tumor_folder_prefix_training)
    # pituitory_tumor = get_object_presigned_url(bucket_name,pituitory_tumor_folder_prefix_training)
    # Print or use the presigned URLs as needed
    # for url in presigned_urls:
    #     print(url)
    print(url)
    return url

extract_data()
