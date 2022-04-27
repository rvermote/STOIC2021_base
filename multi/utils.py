import torch
import pandas as pd
import numpy as np
import config
import cv2
import albumentations as A
import numpy as np
import SimpleITK as sitk
import math as m
from albumentations.pytorch import ToTensorV2
from sklearn import metrics
from tqdm import tqdm
from torch import nn

def make_prediction(model):
    model.eval()
    
    img = sitk.ReadImage('E:/CT_data/data/mha/6.mha')
    #gender = img.GetMetaData(key="PatientSex")
    #age = img.GetMetaData(key="PatientAge")
    

    input_image = sitk.GetArrayFromImage(img)
    
    #get cropped slice from CT scan
    img_slice1 = input_image[m.ceil(0.6*input_image.shape[0]), 115:435, 45:511-46]
    img_slice2 = input_image[m.ceil(0.7*input_image.shape[0]), 115:435, 45:511-46]
    img_slice3 = input_image[m.ceil(0.5*input_image.shape[0]), 115:435, 45:511-46]
    
    #normalize to [0,1]
    clip_min = -1024
    clip_max = 0
    img_slice1 = np.clip(img_slice1, clip_min, clip_max)
    img_slice1 = (img_slice1 - clip_min) / (clip_max - clip_min)
    img_slice2 = np.clip(img_slice2, clip_min, clip_max)
    img_slice2 = (img_slice2 - clip_min) / (clip_max - clip_min)
    img_slice3 = np.clip(img_slice3, clip_min, clip_max)
    img_slice3 = (img_slice3 - clip_min) / (clip_max - clip_min)

    #restore aspect ratio
    s = max(img_slice1.shape[0:2])
    f1 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice1.shape[1])//2,(s - img_slice1.shape[0])//2
    f1[ay:img_slice1.shape[0]+ay,ax:ax+img_slice1.shape[1]] = img_slice1
    
    s = max(img_slice2.shape[0:2])
    f2 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice2.shape[1])//2,(s - img_slice2.shape[0])//2
    f2[ay:img_slice2.shape[0]+ay,ax:ax+img_slice2.shape[1]] = img_slice2
    
    s = max(img_slice3.shape[0:2])
    f3 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice3.shape[1])//2,(s - img_slice3.shape[0])//2
    f3[ay:img_slice3.shape[0]+ay,ax:ax+img_slice3.shape[1]] = img_slice3
    
    input_image1 = cv2.merge([f1,f1,f1])
    input_image2 = cv2.merge([f2,f2,f2])
    input_image3 = cv2.merge([f3,f3,f3])
    
    val_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            ToTensorV2(),
        ],
        additional_targets={'image2': 'image', 'image3': 'image'}
    )
    
    trans = config.val_transforms(image = input_image1, image2=input_image2, image3=input_image3)
    input_image1 = trans["image"]
    input_image2 = trans["image2"]
    input_image3 = trans["image3"]
    input_image1 = input_image1.unsqueeze(0)
    input_image2 = input_image2.unsqueeze(0)
    input_image3 = input_image3.unsqueeze(0)
    input_image1 = input_image1.to(config.DEVICE, dtype=torch.float)
    input_image2 = input_image2.to(config.DEVICE, dtype=torch.float)
    input_image3 = input_image3.to(config.DEVICE, dtype=torch.float)
    with torch.no_grad():
        output = torch.sigmoid(model(input_image1,input_image2,input_image3))
        print(output)
    prob = output.cpu().numpy().astype(float)[0]
    print(prob)

def check_accuracy(model, loader, mode):
    all_preds1, all_labels1, all_preds2, all_labels2, losses = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    num_correct1, num_correct2, num_correct3, num_correct4, num_samples1 = 0,0,0,0,0
    model.eval()
    
    i = 0
    
    for x1, x2, x3, y1, y2, gender, age in tqdm(loader):
        
        x1 = x1.to(config.DEVICE, dtype=torch.float)
        x2 = x2.to(config.DEVICE, dtype=torch.float)
        x3 = x3.to(config.DEVICE, dtype=torch.float)
        gender = gender.to(config.DEVICE, dtype=torch.float)
        age = age.to(config.DEVICE, dtype=torch.float)
        y1 = y1.to(config.DEVICE, dtype=torch.float)
        y2 = y2.to(config.DEVICE, dtype=torch.float)
        with torch.no_grad():
            output = model(x1, x2, x3, gender, age)
            prob = torch.sigmoid(output.cpu())
            
            loss_fn1 = nn.BCEWithLogitsLoss()
            loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2]).cuda())
            if mode == "svr":
                loss = loss_fn1(output[:, :1], y2.unsqueeze(1))
            else:
                loss = loss_fn1(output[:, :1], y1.unsqueeze(1))
            losses = np.append(losses, [loss.item()])
            
            prob = torch.FloatTensor(prob).to(config.DEVICE, dtype=torch.float)
            prob = torch.transpose(prob,0 , 1)
            pred_class = (prob>0.5).long()
        
        if mode == "svr": 
            for j in range(len(pred_class.data[0])-1):
                pointer = int(pred_class.data[0][j].item())
                if pointer == 1 and y2[j].item() == 1:
                    num_correct1 += 1
                elif pointer == 1 and y2[j].item() == 0:
                    num_correct2 += 1
                elif pointer == 0 and y2[j].item() == 0:
                    num_correct3 += 1
                elif pointer == 0 and y2[j].item() == 1:
                    num_correct4 += 1
        else:
            for j in range(len(pred_class.data[0])-1):
                pointer = int(pred_class.data[0][j].item())
                if pointer == 1 and y1[j].item() == 1:
                    num_correct1 += 1
                elif pointer == 1 and y1[j].item() == 0:
                    num_correct2 += 1
                elif pointer == 0 and y1[j].item() == 0:
                    num_correct3 += 1
                elif pointer == 0 and y1[j].item() == 1:
                    num_correct4 += 1
        
        if i==0:
            all_preds1 = prob.detach().cpu().numpy()
            if mode == "svr":
                all_labels1 = y2.detach().cpu().numpy()
            else:
                all_labels1 = y1.detach().cpu().numpy()               
        else:
            all_preds1 = np.append(all_preds1, prob.detach().cpu().numpy())
            if mode == "svr":
                all_labels1 = np.append(all_labels1, y2.detach().cpu().numpy())
            else:
                all_labels1 = np.append(all_labels1, y1.detach().cpu().numpy())
                
        i += 1
    
    auc = metrics.roc_auc_score(all_labels1, all_preds1)
    print(f"auc_severe: {auc}")
    
    #print(f"Got {num_correct1} / {num_samples1} with accuracy {float(num_correct1) / float(num_samples1) * 100:.2f}")
    #print(f"average loss over test set: {sum(losses)/len(losses)}")
    
    #return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(all_labels, axis=0, dtype=np.int64)
    print(num_correct1)
    print(num_correct2)
    print(num_correct3)
    print(num_correct4)
    model.train()
    return sum(losses)/len(losses), auc, num_correct1, num_correct2, num_correct3, num_correct4

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpiont["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr