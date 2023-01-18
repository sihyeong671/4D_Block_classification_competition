import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy

import matplotlib.pyplot as plt

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import transformers

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm


from util import *

import wandb
import warnings
warnings.filterwarnings(action='ignore') 

from inference import inference

from sklearn.model_selection import train_test_split, KFold

CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
]

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    classes_acc = []
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
            class_acc = [labels[i] == preds[i] for i in range(len(labels))]

            val_acc.append(batch_acc)
            val_loss.append(loss.item())
            classes_acc.append(np.mean(class_acc,axis=0))
        
        _classes_acc = np.mean(classes_acc, axis=0)
        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)
    
    return _val_loss, _val_acc, _classes_acc


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

    model = ConvNext_xlarge()
    model.to(device)
    
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingWarmupRestarts(
                                                optimizer=optimizer,
                                                cycle_mult=1,
                                                max_lr=3e-2,
                                                min_lr=1e-5,
                                                warmup_steps=1,
                                                first_cycle_steps=15,
                                                gamma=0.5
                                            )

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
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_acc, _classes_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')
        for _cls, _acc in zip(CLASSES, _classes_acc):
            print(f'{_cls} acc : [{_acc}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = deepcopy(model)
            early_stop = 0
        else:
            early_stop += 1
            
        
        if epoch == 1 or epoch % 100 == 0:
            if args.makecsvfile:
                inference(args=args,model=best_model, epoch=epoch)
            torch.save(best_model, f'./ckpt/{args.model_name}_{args.detail}_{epoch}.pth')
        
        wandb.log({
            "train loss": _train_loss, 
            "val loss": _val_loss,
            "val acc": _val_acc,
            "A Acc": _classes_acc[0],
            "B Acc": _classes_acc[1],
            "C Acc": _classes_acc[2],
            "D Acc": _classes_acc[3],
            "E Acc": _classes_acc[4],
            "F Acc": _classes_acc[5],
            "G Acc": _classes_acc[6],
            "H Acc": _classes_acc[7],
            "I Acc": _classes_acc[8],
            "J Acc": _classes_acc[9],
            "learning rate": scheduler.get_lr()[0]
            })
        
        if early_stop > 7:
            break

    torch.save(best_model, f'./ckpt/{args.model_name}_{args.detail}_{epoch}.pth')
        


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
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--clip', default=1)
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
    
