from datavisualization import visualise_image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import requests
from PIL import Image
from io import BytesIO
import torch
import dill as pickle
import numpy as np
import glob
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
import torch

def transform_data():
    url = visualise_image()
    ALL_CLASSES = ['background', 'waterbody']

    label_map = [
        (0, 0, 0), # Background.
        (255, 255, 255), # Waterbody.
    ]
    
    def set_class_values(all_classes, classes_to_train):
        class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
        return class_values
    
    def get_label_mask(mask, class_values, label_colors_list):
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for value in class_values:
            for ii, label in enumerate(label_colors_list):
                if value == label_colors_list.index(label):
                    label = np.array(label)
                    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
        label_mask = label_mask.astype(int)
        return label_mask
    
    def get_images(root_path):
        train_images = glob.glob(f"{root_path}/train_images/*")
        train_images.sort()
        train_masks = glob.glob(f"{root_path}/train_masks/*")
        train_masks.sort()
        valid_images = glob.glob(f"{root_path}/valid_images/*")
        valid_images.sort()
        valid_masks = glob.glob(f"{root_path}/valid_masks/*")
        valid_masks.sort()
        return train_images , train_masks , valid_images, valid_masks
    
    def normalize():
        transform = A.Compose([
            A.Normalize(
                mean=[0.45734706, 0.43338275, 0.40058118],
                std=[0.23965294, 0.23532275, 0.2398498],
                always_apply=True
            )
        ])
        return transform
    
    def train_transforms(img_size):
        train_image_transform = A.Compose([
            A.Resize(img_size, img_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        return train_image_transform
    
    def valid_transforms(img_size):
        valid_image_transform = A.Compose([
            A.Resize(img_size, img_size, always_apply=True),
        ])
        return valid_image_transform
    
    class SegmentationDataset(Dataset):
        
        def __init__(self, image_paths, mask_paths, tfms, norm_tfms, label_colors_list, classes_to_train, all_classes):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.tfms = tfms
            self.norm_tfms = norm_tfms
            self.label_colors_list = label_colors_list
            self.all_classes = all_classes
            self.classes_to_train = classes_to_train
            # Convert string names to class values for masks.
            self.class_values = set_class_values(self.all_classes, self.classes_to_train)
    
        def __len__(self):
            return len(self.image_paths)
    
        def __getitem__(self, index):
            image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
            mask = np.array(Image.open(self.mask_paths[index]).convert('RGB'))
    
            # Make any pixel value above 200 as 255 for waterbody.
            im = mask >= 200
            mask[im] = 255
            mask[np.logical_not(im)] = 0
    
            image = self.norm_tfms(image=image)['image']
            transformed = self.tfms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # Get colored label mask.
            mask = get_label_mask(mask, self.class_values, self.label_colors_list)
           
            image = np.transpose(image, (2, 0, 1))
            
            image = torch.tensor(image, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.long) 
    
            return image, mask
    
    def get_dataset(train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths, all_classes, classes_to_train, label_colors_list, img_size):
        train_tfms = train_transforms(img_size)
        valid_tfms = valid_transforms(img_size)
        norm_tfms = normalize()
    
        train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, train_tfms, norm_tfms, label_colors_list, classes_to_train, all_classes)
        valid_dataset = SegmentationDataset(valid_image_paths, valid_mask_paths, valid_tfms, norm_tfms, label_colors_list, classes_to_train, all_classes)
        return train_dataset, valid_dataset
    
    train_images, train_masks, valid_images, valid_masks = get_images(os.path.join(os.getcwd(),"Water_Bodies_Dataset_Split"))    

    classes_to_train = ALL_CLASSES
    
    train_dataset, valid_dataset = get_dataset(train_images, train_masks, valid_images, valid_masks, ALL_CLASSES, classes_to_train, label_map, img_size=224)
    
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)
    return train_dataset,valid_dataset
transform_data()
