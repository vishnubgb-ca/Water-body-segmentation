from extract_data import extract_data
from PIL import Image
import requests
from io import BytesIO
import os 
import pathlib 
import zipfile
import random

def open_random_image(path):
    # Get a list of all files in the folder
    all_files = os.listdir(path)
    random_image_file = random.choice(all_files)
    image_path = os.path.join(path, random_image_file)
    image = Image.open(image_path)
    return image
    

def visualise_image():
    url = extract_data()
    url_response = requests.get(url)
    with zipfile.ZipFile(BytesIO(url_response.content)) as z:
        z.extractall('.')
    print(os.path.join(os.getcwd(),"Training/glioma_tumor"),"------------",os.listdir(os.path.join(os.getcwd(),"Training/glioma_tumor")))
    path = pathlib.Path(os.path.join(os.getcwd(),"Training"))
    # glioma_tumor_image_urls,meningioma_tumor_image_urls,no_tumor_image_urls,pituitory_tumor_image_urls = extract_data()
    # glioma_tumor_response = requests.get(glioma_tumor_image_urls[0])
    # meningioma_tumor_response = requests.get(meningioma_tumor_image_urls[0])
    # no_tumor_image_response = requests.get(no_tumor_image_urls[0])
    # pituitory_tumor_image_response = requests.get(pituitory_tumor_image_urls[0])
    
    glioma_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/glioma_tumor"))
    meningioma_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/meningioma_tumor"))
    no_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/no_tumor"))
    pituitory_tumor_image = open_random_image(os.path.join(os.getcwd(),"Training/pituitory_tumor"))
    glioma_tumor_image.save('glioma_tumor.jpg')
    meningioma_tumor_image.save('meningioma_tumor.jpg')
    no_tumor_image.save('no_tumor.jpg')
    pituitory_tumor_image.save('pituitory_tumor.jpg')
    #print(glioma_tumor_image.show())
    #print(meningioma_tumor_image.show())
    #print(no_tumor_image.show())
    #print(pituitory_tumor_image.show())
    return path

visualise_image()
