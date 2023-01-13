import random
import pandas as pd
import numpy as np
import os
import cv2
import argparse

import transformers
from transformers import ConvNextForImageClassification

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm

from util import *

import warnings
warnings.filterwarnings(action='ignore')


def infer(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            probs = model(imgs)
            probs  = probs.cpu().detach().numpy()
            # soft voting시 여기 고치면 됨
            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return predictions


def inference(args):

    seed_everything(args.seed)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    test_transform = A.Compose([
                                A.Resize(args.img_size, args.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test = pd.read_csv('./data/test.csv')
    test['img_path'] = test['img_path'].apply(lambda x: './data'+x[1:])
    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = torch.load(f"./ckpt/{args.model_name}_{args.detail}_{args.epoch}.pth")

    preds = infer(model, test_loader, device)
    submit = pd.read_csv('./data/sample_submission.csv')
    submit.iloc[:,1:] = preds
    submit.to_csv(f'./{args.model_name}_{args.detail}_{args.epoch}.csv', index=False)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', default="ConvNext")
    parser.add_argument('--detail', default="xlarge_384")
    args = parser.parse_args()
    
    
    inference(args)
