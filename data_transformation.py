from datavisualization import visualise_image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import requests
from PIL import Image
from io import BytesIO
import torch
import dill as pickle

def transform_data():
    path = visualise_image()
    data_transform = torchvision.transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # tumor_list = [glioma_tumor_image_urls,meningioma_tumor_image_urls,no_tumor_image_urls,pituitory_tumor_image_urls]
    # images = []
    # for lst in tumor_list:
    #     for url in lst:
    #         response = requests.get(url)
    #         image = Image.open(BytesIO(response.content))
    #         images.append(image)
    
    # transformed_images = [data_transform(img) for img in images]
    # model_dataset = torch.stack(transformed_images)
    

    # model_dataset = datasets.ImageFolder(path, transform=data_transform)

    # def download_image(url):
    #     response = requests.get(url)
    #     img = Image.open(BytesIO(response.content))
    #     return img
    
    # class CustomImageDataset(Dataset):
    #     def __init__(self, image_urls, transform=None):
    #         self.image_urls = image_urls
    #         self.transform = transform

    #     def __len__(self):
    #         return len(self.image_urls)

    #     def __getitem__(self, idx):
    #         img = download_image(self.image_urls[idx])
    #         if self.transform:
    #             img = self.transform(img)
    #         return img
    
    # all_image_urls = glioma_tumor_image_urls+meningioma_tumor_image_urls+no_tumor_image_urls+pituitory_tumor_image_urls
    # model_dataset = CustomImageDataset(all_image_urls, transform=data_transform)
    # torch.save(model_dataset,'model_dataset.pt')
    model_dataset = datasets.ImageFolder(path, transform=data_transform)
    with open('model_dataset.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset
transform_data()
