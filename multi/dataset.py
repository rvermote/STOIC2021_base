import config
import os
import cv2
import pandas as pd
import numpy as np
import math as m
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class CTDataset(Dataset):
    def __init__(self, image_folder1,image_folder2,image_folder3, path_to_label, path_to_meta, train=True, transform=None):
        super().__init__
        self.label_file = pd.read_csv(path_to_label)
        self.meta_file = pd.read_csv(path_to_meta)
        self.image_folder1 = image_folder1
        self.image_files1 = os.listdir(image_folder1)
        self.image_folder2 = image_folder2
        self.image_files2 = os.listdir(image_folder2)
        self.image_folder3 = image_folder3
        self.image_files3 = os.listdir(image_folder3)
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.image_files1)
    
    def __getitem__(self, index):
        image_file1 = self.image_files1[index]
        image1 = np.genfromtxt(os.path.join(self.image_folder1, image_file1), delimiter=',')
        image1 = cv2.merge([image1,image1,image1])
        
        image_file2 = self.image_files2[index]
        image2 = np.genfromtxt(os.path.join(self.image_folder2, image_file2), delimiter=',')
        image2 = cv2.merge([image2,image2,image2])
        
        image_file3 = self.image_files3[index]
        image3 = np.genfromtxt(os.path.join(self.image_folder3, image_file3), delimiter=',')
        image3 = cv2.merge([image3,image3,image3])
        
        ID_row = self.label_file.loc[self.label_file['PatientID'] == int(image_file1.replace('.csv',''))]
        lbl_cvd = int(ID_row.iloc[:,1])
        lbl_svr = int(ID_row.iloc[:,2])
        
        ID_row = self.meta_file.loc[self.meta_file['PatientID'] == int(image_file1.replace('.csv',''))]
        gender = float(ID_row.iloc[:,1])
        age = float(ID_row.iloc[:,2])
        
        if index in self.transform:
            trans = config.val_transforms(image = image1, image2=image2, image3=image3)
            image1 = trans["image"]
            image2 = trans["image2"]
            image3 = trans["image3"]
        else:
            trans = config.train_transforms(image = image1, image2=image2, image3=image3)
            image1 = trans["image"]
            image2 = trans["image2"]
            image3 = trans["image3"]
            
        return image1, image2, image3, lbl_cvd, lbl_svr, gender, age