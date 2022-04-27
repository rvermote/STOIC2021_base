import os
import SimpleITK as sitk
import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def resample(itk_image,
             new_spacing,
             outside_val: float = 0
             ) -> sitk.Image:

    shape = itk_image.GetSize()
    spacing = itk_image.GetSpacing()
    output_shape = tuple(int(round(s * os / ns)) for s, os, ns in zip(shape, spacing, new_spacing))
    return sitk.Resample(
        itk_image,
        output_shape,
        sitk.Transform(),
        sitk.sitkLinear,
        itk_image.GetOrigin(),
        new_spacing,
        itk_image.GetDirection(),
        outside_val,
        sitk.sitkFloat32,
    )

ID, gender, age = np.array([]), np.array([]), np.array([])

for image in tqdm(os.listdir('E:/CT_data/data/mha')):
    
    img = sitk.ReadImage(os.path.join('E:/CT_data/data/mha/',image))
    single_ID = image.replace(".mha","")
    ID = np.append(ID, single_ID)
    
    path_to_label = 'E:/CT_data/metadata/reference.csv'
    label_file = pd.read_csv(path_to_label)
    ID_row = label_file.loc[label_file['PatientID'] == int(single_ID)]
    lbl_cvd = int(ID_row.iloc[:,1])
    #img = resample(img, new_spacing=(1.6, 1.6, 1.6))

    
    #try:
    #    if img.GetMetaData(key="PatientSex") == "M":
    #        gender = np.append(gender, [1])
    #    else:
    #        gender = np.append(gender, [0])
    #except:
    #    gender = np.append(gender, [0])
    #
    #try:
    #    if img.GetMetaData(key="PatientAge"):
    #        age = np.append(age, [float(img.GetMetaData(key="PatientAge"))/100])
    #    else:
    #        age = np.append(age, [0.55])
    #except:
    #    age = np.append(age, [0.55])
    
    if 1==1:
    #if lbl_cvd == 1:
    
        img_data = sitk.GetArrayFromImage(img)
        
        plt.figure()
        plt.imshow(img_data[m.ceil(0.6*img_data.shape[0]), :, :])
        plt.show()

        img_slice = img_data[m.ceil(0.6*img_data.shape[0]), 115:435, 45:511-46]
        img_slice2 = img_data[m.ceil(0.65*img_data.shape[0]), 115:435, 45:511-46]
        img_slice3 = img_data[m.ceil(0.55*img_data.shape[0]), 115:435, 45:511-46]
        
        plt.figure()
        plt.imshow(img_slice)
        plt.show()
    
        #clip and normalize to [0,1]
        clip_min = -1024
        clip_max = 0
    
        img_slice = np.clip(img_slice, clip_min, clip_max)
        img_slice = (img_slice - clip_min) / (clip_max - clip_min)
        img_slice2 = np.clip(img_slice2, clip_min, clip_max)
        img_slice2 = (img_slice2 - clip_min) / (clip_max - clip_min)
        img_slice3 = np.clip(img_slice3, clip_min, clip_max)
        img_slice3 = (img_slice3 - clip_min) / (clip_max - clip_min)

        #restore aspect ratio
        s = max(img_slice.shape[0:2])
        f = np.zeros((s,s),np.float32)
        ax,ay = (s - img_slice.shape[1])//2,(s - img_slice.shape[0])//2
        f[ay:img_slice.shape[0]+ay,ax:ax+img_slice.shape[1]] = img_slice
        s = max(img_slice2.shape[0:2])
        f2 = np.zeros((s,s),np.float32)
        ax,ay = (s - img_slice2.shape[1])//2,(s - img_slice2.shape[0])//2
        f2[ay:img_slice2.shape[0]+ay,ax:ax+img_slice2.shape[1]] = img_slice2
        s = max(img_slice3.shape[0:2])
        f3= np.zeros((s,s),np.float32)
        ax,ay = (s - img_slice3.shape[1])//2,(s - img_slice3.shape[0])//2
        f3[ay:img_slice3.shape[0]+ay,ax:ax+img_slice3.shape[1]] = img_slice3
        
        plt.figure()
        plt.imshow(f)
        plt.show()
    
        #np.savetxt(os.path.join("C:/Users/Robin/Documents/data/preprocess_0.6_0clip_covidonly",image.replace(".mha",".csv")), f1, delimiter=",")
        #np.savetxt(os.path.join("E:/CT_data/saved/preprocess_0.65_0clip",image.replace(".mha",".csv")), f2, delimiter=",")
        #np.savetxt(os.path.join("E:/CT_data/saved/preprocess_0.55_0clip",image.replace(".mha",".csv")), f3, delimiter=",")
    
#df = pd.DataFrame([ID, gender, age])
#df = df.T   

#df.to_csv("C:/Users/Robin/Documents/metadata/extraMeta.csv", header=False, index=False)