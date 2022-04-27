import torch
from torch import nn, optim
from numpy import sqrt 
import numpy as np
import pandas as pd
import os
import config
import torchvision
import gc
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from dataset import CTDataset
from torchvision.utils import save_image
from utils import(load_checkpoint, save_checkpoint, check_accuracy, make_prediction)
from sklearn.model_selection import KFold
from operator import truediv
import matplotlib.pyplot as plt

mode = "svr"

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.cnn1 = torchvision.models.vgg19(pretrained=True)
        self.cnn2 = torchvision.models.vgg19(pretrained=True)
        self.cnn3 = torchvision.models.vgg19(pretrained=True)
        
        self.fc1 = nn.Linear(32+32+32, 32)
        self.fc2 = nn.Linear(32, 1)
        
        for model in [self.cnn1, self.cnn2, self.cnn3]:
            for p in model.parameters():
                p.requires_grad = False
            #counter=0
            #for name,p in model.named_parameters():
            #    counter+=1
            #    if counter > 26:
            #        p.requires_grad = True
            model.classifier[6] = nn.Sequential(nn.Linear(model.classifier[6].in_features, 32))
                                                
    
    
    def forward(self, image1, image2, image3, gender=None, age=None):
        x1 = self.cnn1(image1)
        x2 = self.cnn2(image2)
        x3 = self.cnn3(image3)
        #x2 = gender.reshape(-1,1)
        #x3 = age.reshape(-1,1)
        
        x = torch.cat((x1,x2,x3), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses=[]
    loop = tqdm(loader)
    for data1,data2,data3,target1,target2,gender,age in loop:

        data1 = data1.to(device=device, dtype=torch.float)
        data2 = data2.to(device=device, dtype=torch.float)
        data3 = data3.to(device=device, dtype=torch.float)
        gender = gender.to(device=device, dtype=torch.float)
        age = age.to(device=device, dtype=torch.float)
        target1 = target1.to(device=device, dtype=torch.float)
        target2 = target2.to(device=device, dtype=torch.float)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(data1, data2 ,data3, gender, age)
            if mode=="svr":
                loss = loss_fn(output[:, :1], target2.unsqueeze(1))
            else:
                loss = loss_fn(output[:, :1], target1.unsqueeze(1))
        
        losses.append(loss.item())

        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())
        
    print(f"Loss average over epoch: {sum(losses)/len(losses)}")
    
    return sum(losses)/len(losses)
    
def main():
    
    #torch.cuda.set_per_process_memory_fraction(0.9, 0)
    
    if mode=="svr":
        full_ds = CTDataset(
            image_folder1 = 'E:/CT_data/saved/preprocess_0.7_0clip_covidonly',
            image_folder2 = 'E:/CT_data/saved/preprocess_0.6_0clip_covidonly',
            image_folder3 = 'E:/CT_data/saved/preprocess_0.5_0clip_covidonly',
            path_to_label = 'E:/CT_data/metadata/reference.csv',
            path_to_meta = 'E:/CT_data/saved/extraMeta.csv'
        )
    else:
        full_ds = CTDataset(
            image_folder1 = 'E:/CT_data/saved/preprocess_0.7_0clip_train-full',
            image_folder2 = 'E:/CT_data/saved/preprocess_0.6_0clip_train-full',
            image_folder3 = 'E:/CT_data/saved/preprocess_0.5_0clip_train-full',
            path_to_label = 'E:/CT_data/metadata/reference.csv',
            path_to_meta = 'E:/CT_data/saved/extraMeta.csv'
        )
    
    #train_ds = CTDataset(
    #    image_folder1 = 'C:/Users/Robin/Documents/data/preprocess_0.6_0clip_train',
    #    image_folder2 = 'C:/Users/Robin/Documents/data/preprocess_0.5_0clip_train',
    #    image_folder3 = 'C:/Users/Robin/Documents/data/preprocess_0.7_0clip_train',
    #    path_to_label = 'C:/Users/Robin/Documents/metadata/reference.csv',
    #    path_to_meta = 'C:/Users/Robin/Documents/metadata/extraMeta.csv',
    #    transform = config.train_transforms, 
    #)
    #test_ds = CTDataset(
    #    image_folder1 = 'C:/Users/Robin/Documents/data/preprocess_0.6_0clip_test',
    #    image_folder2 = 'C:/Users/Robin/Documents/data/preprocess_0.5_0clip_test',
    #    image_folder3 = 'C:/Users/Robin/Documents/data/preprocess_0.7_0clip_test',
    #    path_to_label = 'C:/Users/Robin/Documents/metadata/reference.csv',
    #    path_to_meta = 'C:/Users/Robin/Documents/metadata/extraMeta.csv',
    #    transform = config.val_transforms, 
    #    train = False
    #)
    
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    #kf = KFold(n_splits=config.NUM_SPLITS)
    
    all_auc, all_sen, all_spec, all_NPV, all_PPV, all_f1 = [], [], [], [], [], []
    tot_TP, tot_FP, tot_TN, tot_FN = 0, 0, 0, 0
    
    for fold,(train_idx,test_idx) in enumerate(KFold(n_splits=config.NUM_SPLITS).split(full_ds)):
        
        if 1 == 1:
            if mode=="svr":
                full_ds = CTDataset(
                    image_folder1 = 'E:/CT_data/saved/preprocess_0.7_0clip_covidonly',
                    image_folder2 = 'E:/CT_data/saved/preprocess_0.6_0clip_covidonly',
                    image_folder3 = 'E:/CT_data/saved/preprocess_0.5_0clip_covidonly',
                    path_to_label = 'E:/CT_data/metadata/reference.csv',
                    path_to_meta = 'E:/CT_data/saved/extraMeta.csv',
                    transform = test_idx, 
                )
            else:
                full_ds = CTDataset(
                    image_folder1 = 'E:/CT_data/saved/preprocess_0.7_0clip_train-full',
                    image_folder2 = 'E:/CT_data/saved/preprocess_0.6_0clip_train-full',
                    image_folder3 = 'E:/CT_data/saved/preprocess_0.5_0clip_train-full',
                    path_to_label = 'E:/CT_data/metadata/reference.csv',
                    path_to_meta = 'E:/CT_data/saved/extraMeta.csv',
                    transform = test_idx, 
                )

            gc.collect(generation=0)
            gc.collect(generation=1)
            gc.collect(generation=2)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            train_loader = DataLoader(
                full_ds,
                batch_size = config.BATCH_SIZE,
                num_workers = config.NUM_WORKERS,
                pin_memory = config.PIN_MEMORY,
                sampler = train_subsampler
            )
            test_loader = DataLoader(
                full_ds,
                batch_size = config.BATCH_SIZE,
                num_workers = config.NUM_WORKERS,
                sampler = test_subsampler
            )

            model = MyModel()
            model = model.to(config.DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            optimizer_finetune = optim.Adam(model.parameters(), lr=config.FINETUNE_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)     

            train_losses, avg_losses, avg_auc, TP, FP, TN, FN = [], [], [], [], [], [], []
            
            for epoch in range(config.NUM_PRETRAIN_EPOCHS):

                loss = train_one_epoch(train_loader, model, optimizer_finetune, loss_fn, scaler, config.DEVICE)
                result = check_accuracy(model, test_loader, mode)

                train_losses.append(loss)
                avg_losses.append(result[0])
                avg_auc.append(result[1])
                TP.append(result[2])
                FP.append(result[3])
                TN.append(result[4])
                FN.append(result[5])

            #for network in [model.cnn1, model.cnn2]:

        #     counter=0
          #      for name,p in network.named_parameters():
           #         counter+=1
            #        if counter > 24:
             #           p.requires_grad = True

            for epoch in range(config.NUM_EPOCHS):

                loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
                result = check_accuracy(model, test_loader, mode)

                train_losses.append(loss)
                avg_losses.append(result[0])
                avg_auc.append(result[1])
                TP.append(result[2])
                FP.append(result[3])
                TN.append(result[4])
                FN.append(result[5])

            #check_accuracy(model, test_loader)
            x = np.arange(start=1, stop=config.NUM_EPOCHS+config.NUM_PRETRAIN_EPOCHS+1)
            plt.plot(x, train_losses, linestyle='--', label="train")
            plt.plot(x, avg_losses, linestyle='--', label="test")
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('average loss')
            plt.show()
            plt.plot(x, avg_auc, linestyle='--')
            plt.ylabel('AUC (ROC)')
            plt.show()
            sens = list(map(np.divide, TP, [x + y for x, y in zip(TP, FN)]))
            plt.plot(x, sens, linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('sensitivity (recall; TPR)')
            plt.show()
            spec = list(map(np.divide, TN, [x + y for x, y in zip(TN, FP)]))
            plt.plot(x, spec, linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('specificity (selectivity; TNR)')
            plt.show()
            PPV = list(map(np.divide, TP, [x + y for x, y in zip(TP, FP)]))
            plt.plot(x, PPV, linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('precision (PPV)')
            plt.show()
            NPV = list(map(np.divide, TN, [x + y for x, y in zip(TN, FN)]))
            plt.plot(x, NPV, linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('precision (NPV)')
            plt.show()
            f1 = list(map(np.divide, 2*TP, [x + y + z for x, y, z in zip(2*TP,FP,FN)]))
            plt.plot(x, f1, linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('F1 Score')
            plt.show()

            all_auc.append(avg_auc)
            all_sen.append(sens)
            all_spec.append(spec)
            all_NPV.append(NPV)
            all_PPV.append(PPV)
            all_f1.append(f1)
            
            tot_TP += TP[len(TP)-1]
            tot_FP += FP[len(FP)-1]
            tot_TN += TN[len(TN)-1]
            tot_FN += FN[len(FN)-1]

            #state = {'model': MyModel(),
            #         'state_dict': model.state_dict()}
            #torch.save(state, os.path.join(os.getcwd(),"model_pars_final_cvd.pth"))
            
    print(tot_TP)
    print(tot_FP)
    print(tot_TN)
    print(tot_FN)
        
    x = np.arange(start=1, stop=config.NUM_EPOCHS+config.NUM_PRETRAIN_EPOCHS+1)
    
    for i in range(len(all_auc)):
        plt.plot(x, all_auc[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.show()
    
    #var
    meanauc = 0
    for auc in all_auc:
        meanauc += auc[len(auc)-1]
    print(len(all_auc))
    meanauc = meanauc/len(all_auc)
    print(meanauc)
    aucVar = meanauc*(1-meanauc)/min(tot_TN+tot_FP, tot_TP+tot_FN)
    print(aucVar)
    #ci
    print(meanauc+1.96*sqrt(aucVar))
    print(meanauc-1.96*sqrt(aucVar))
    
    for i in range(len(all_sen)):
        plt.plot(x, all_sen[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('sensitivity (recall; TPR)')
    plt.show()
    
    #var
    Sn = tot_TP/(tot_TP+tot_FN)
    print(Sn*(1-Sn)/(tot_TP+tot_FN))
    
    #ci
    print(tot_TP)
    print(tot_FN)
    bt = binomtest(k=tot_TP, n=tot_TP+tot_FN)
    print(bt.proportion_ci(method="exact"))
    
    for i in  range(len(all_spec)):
        plt.plot(x, all_spec[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('specificity (selectivity; TNR)')
    plt.show()
    
    #var
    Sp = tot_TN/(tot_TN+tot_FP)
    print(Sp*(1-Sp)/(tot_TN+tot_FP))
    
    #ci
    bt = binomtest(k=tot_TN, n=tot_TN+tot_FP)
    print(bt.proportion_ci(method="exact"))
    
    for i in  range(len(all_PPV)):
        plt.plot(x, all_PPV[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision (PPV)')
    plt.show()
    
    #var
    p = (tot_TP+tot_FN)/(tot_TP+tot_FN+tot_TN+tot_FP)
    varPPV = ( ( pow(p*(1-Sp)*(1-p),2) * Sn*(1-Sn)/(tot_TP+tot_FN) ) + ( pow((Sn)*(1-p)*p,2) * Sp*(1-Sp)/(tot_TN+tot_FP) ) ) / pow((Sn*p) + ((1-Sp) * (1-p)),4)
    print(varPPV)
    
    #ci
    print(1.96*sqrt(varPPV))
    print((tot_TN/(tot_TN+tot_FN)) + 1.96*sqrt(varPPV))
    
    for i in range(len(all_NPV)):
        plt.plot(x, all_NPV[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision (NPV)')
    plt.show()
        
    #var
    p = (tot_TP+tot_FN)/(tot_TP+tot_FN+tot_TN+tot_FP)
    varNPV = ( ( pow(Sp*(1-p)*p,2) * (Sn*(1-Sn))/(tot_TP+tot_FN) ) + ( pow((1-Sn)*(1-p)*p,2) * (Sp*(1-Sp))/(tot_TN+tot_FP) ) ) / pow(((1-Sn)*p) + (Sp * (1-p)),4)
    print(varNPV)
    
    #ci
    print(1.96*sqrt(varNPV))
    print((tot_TN/(tot_TN+tot_FN)) + 1.96*sqrt(varNPV))
    
    for i in range(len(all_f1)):
        plt.plot(x, all_f1[i], linestyle='--', label="split "+str(i+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('F1 Score')
    plt.show()
    
    #model.load_state_dict(torch.load(os.path.join(os.getcwd(),"model_pars_multi3.pth")))
    
    #torch.save(model, os.path.join(os.getcwd(),"model_architecture.pth"))
    
    #save final model parameters
    #torch.save(model.state_dict(), os.path.join(os.getcwd(),"model_pars_multi3_final.pth"))
    
if __name__ == "__main__":
    main()

#if __name__ == "__main__":

#    load_model = torch.load("E:/CT_data/stoic_tl/algorithm/model_cvd.pth", map_location=torch.device(config.DEVICE))
#    model = load_model["model"]
#    model = model.to(config.DEVICE)
#    model.load_state_dict(load_model["state_dict"])

#    make_prediction(model)