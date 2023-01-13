import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

data_dir = "../data/train"
background_dir = "../data/crawed_img/child/"
background_dir_2 = "../data/crawed_img/indoor/"

img_list = sorted(glob(os.path.join(data_dir, "*.jpg")))
background_list = glob(os.path.join(background_dir,"*.jpg"))
background_list_2 = glob(os.path.join(background_dir_2,"*.jpg"))

## background_list, img_list 개수 맞추기
background_list += background_list_2 + background_list_2 + background_list_2 + background_list_2[:2635]

for img_l, background in tqdm(zip(img_list,background_list)):
        # './train\\TRAIN_00999.jpg'
        name = img_l[-15:]
        img = cv2.imread(img_l)

        thresh = 230
        
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

        cv2.imwrite(f"../data/merge_train/{name}",img_filter)