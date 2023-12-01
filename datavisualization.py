from extract_data import extract_data
from PIL import Image
import requests
from io import BytesIO
import os 
import pathlib 
import zipfile
import random

def open_random_images(path):
    # Get a list of all files in the folder
    all_files = os.listdir(path)
    random.shuffle(all_files)
    return all_files[:4]
    

def visualise_image():
    url = extract_data()
    url_response = requests.get(url)
    with zipfile.ZipFile(BytesIO(url_response.content)) as z:
        z.extractall('.')
    images = open_random_images(os.path.join(os.getcwd(),"Water_Bodies_Dataset_Split/train_images"))
    for i in range(4):
        images[i].save('sample'+str(i)+'.jpg')
    #meningioma_tumor_image.save('meningioma_tumor.jpg')
    return url

visualise_image()
