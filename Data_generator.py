import torch
import os
from os import listdir
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns

import cv2
from matplotlib.image import imread

# import tensorflow as tf
# from keras.utils.np_utils import to_categorical
# from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import glob
import PIL
import random

random.seed(100)
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import os
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.transform import resize
import pandas as pd
import torchvision.transforms.functional as TF

def imshow(img, title=None):
    img = img.detach().cpu().numpy()  # Convert to NumPy
    img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


def maskshow(img, title=None):
    img = img.detach().cpu().numpy() # Convert to NumPy
    #img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

class SegmentationTransform_test:
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size
    def __call__(self, image, mask):
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)  # keep mask integers

        # Convert to tensor if not already
        if not torch.is_tensor(image):
            image = TF.to_tensor(image)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)

        # Normalize image only
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        return image, mask

class SegmentationTransform:
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size

    def __call__(self, image, mask):
        Temp = random.randint(0, 3)
        # Random horizontal flip
        if Temp == 1:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        elif Temp == 2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        elif Temp == 3 : 
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Resize to fixed size
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)  # keep mask integers

        # Convert to tensor if not already
        if not torch.is_tensor(image):
            image = TF.to_tensor(image)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)

        # Normalize image only
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        return image, mask

class CBISDDSMDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, img_size=(512, 512)):
        self.mass_data = pd.read_csv(csv_path)
        #self.mass_data = self.mass_data[self.mass_data["img_path"].str.contains("Mass", na=False)]
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.mass_data)  
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.mass_data.iloc[idx]['img_path'])
        mask_path = os.path.join(self.root_dir, self.mass_data.iloc[idx]['roi_path'])
        label = self.mass_data.iloc[idx]['label']

        image = pydicom.read_file(img_path).pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        mask = pydicom.read_file(mask_path).pixel_array
        mask = (mask > 0).astype(np.float32)
        if label == 4:
            mask = np.where(mask == 1, 2, mask)
        elif label == 2 :
            mask = np.where(mask == 1, 3, mask)


        image = np.expand_dims(image, axis=0)  # [1, H, W]
        mask = np.expand_dims(mask, axis=0)    # [1, H, W]

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Apply paired transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        mask = mask.squeeze(0)
        return image, mask

# # Example usage
if __name__ == "__main__":
    paired_transform = SegmentationTransform(img_size=(512, 512))

   
    
    
    dataset = CBISDDSMDataset(
    csv_path="training_dataset.csv",
    root_dir="",
    transform=paired_transform,
    img_size=(512, 512)
)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Example batch
    batch = next(iter(dataloader))
    images, masks = batch['image'], batch['mask']
    print(f"Images shape: {images.shape}")  # Should be [batch_size, 1, 512, 512]
    print(f"Masks shape: {masks.shape}")    # Should be [batch_size, 1, 512, 512]