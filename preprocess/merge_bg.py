import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import imutils
from random import randint

data_dir = "../data/train"
bg_dir = "../data/crawed_img"

os.makedirs('../data/merge_train', exist_ok=True)

keywords = ['indoor', 'table', 'child']

img_list = sorted(glob(os.path.join(data_dir, "*.jpg")))

bg_list = []
for k in keywords:
    bg_list.extend(glob(os.path.join(bg_dir, k, "*.jpg")))

for img_l, background in tqdm(zip(img_list, bg_list)):
    # './train\\TRAIN_00999.jpg'
    name = img_l[-15:]
    img = cv2.imread(img_l)

    thresh = 230
    
    if np.random.uniform() > 0.5: # block size 줄이기 + 이동
        h, w, c = img.shape
        random_size = randint(50, 100)
        random_h = randint(-80, 80) 
        random_w = randint(-80, 80)
        resize_img= cv2.resize(
            cv2.copyMakeBorder(
                img, 
                random_size,
                random_size,
                random_size,
                random_size,
                cv2.BORDER_CONSTANT,value=(255, 255, 255)), (h, w))
        move_img = imutils.translate(resize_img, random_w, random_h)
        move_img[np.where(move_img == 0)] = 255
        img = move_img
    else: # block size 늘리기
        random_size = randint(450, 550) 
        resize_img = cv2.resize(img, (random_size, random_size))
        crop_img = img[50:450, 50:450, :]
        img = crop_img
    
    img_bin = np.array(img[:, :, 0] < thresh, dtype=np.uint8) * 255
    img_blur = cv2.medianBlur(img_bin, 5)
    img_filter = img.copy()
    img_filter[img_blur == 0] = 0

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img_blur)
    stats = np.array(stats)
    stats[0, -1] = 0
    label_idx = np.argmax(stats[:, -1])
    img_mask = labels == label_idx
    img_final = img_filter.copy()
    img_final[~img_mask] = (0, 0, 0)

    img_back = cv2.resize(cv2.imread(background), img.shape[::-1][1:])
    img_back[img_mask] = (0, 0, 0)
    img_filter += img_back

    
    with open('./used_name.txt', 'a') as f:
        f.write(f'{background}\n')
    cv2.imwrite(f"../data/merge_train/{name}",img_filter)