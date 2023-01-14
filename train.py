import pandas as pd
import numpy as np
import os

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm

import argparse

from util import *

import wandb
import warnings
warnings.filterwarnings(action='ignore') 

from inference import inference


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            probs = model(imgs)
            
            loss = criterion(probs, labels)
            
            probs  = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            
            preds = probs > 0.5
            batch_acc = (labels == preds).mean()
            
            val_acc.append(batch_acc)
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)
    
    return _val_loss, _val_acc


def train(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    early_stop = 0
    
    ## 데이터 셋 설정
    df = pd.read_csv('./data/train.csv')
    df["img_path"] = df['img_path'].apply(lambda x: './data/merge_'+x[2:]) # img path 변경
    df = df.sample(frac=1)
    
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    val_df = df[train_len:]

    train_labels = get_labels(train_df)
    val_labels = get_labels(val_df)

    
    train_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                        ])
    
    test_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                        ])
    
    train_dataset = CustomDataset(train_df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = CustomDataset(val_df['img_path'].values, val_labels, test_transform)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvNextv2_large()
    model.to(device)
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)


    best_val_acc = 0
    best_model = None
    
    criterion = nn.BCELoss().to(device)
    
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')
        
        if scheduler is not None:
            scheduler.step(_val_acc)
            
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = deepcopy(model)
        else:
            early_stop += 1
            
        
        if epoch == 1 or epoch % 20 == 0:
            if args.makecsvfile:
                inference(args, best_model, epoch)
                
        wandb.log({
            "train loss": _train_loss, 
            "val loss": _val_loss,
            "val acc": _val_acc
            })
        
        if early_stop > 5:
            torch.save(best_model, f'./ckpt/{args.model_name}_{args.detail}_{epoch}.pth')
            break
            

    torch.save(best_model, f'./ckpt/{args.model_name}_{args.detail}_{args.epochs}.pth')
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', default="ConvNext")
    parser.add_argument('--detail', default="xlarge_384")
    parser.add_argument('--makecsvfile', type=bool ,default=False)
    # parser.add_argument('--checkpoints', default="microsoft/beit-base-patch16-224-pt22k-ft22k")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    wandb.init(
        entity="dacon_4dblock",
        project=args.model_name,
        name=args.detail,
        config={"epochs": args.epochs, "batch_size": args.batch_size}
    )
    
    train(args)
    
