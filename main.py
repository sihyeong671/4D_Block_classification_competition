import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os

st.set_page_config(layout='wide')

st.title('포디 블럭 구조 추출 데이터')

DATA_PATH = './data'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

@st.experimental_memo
def set_data(train_data_path: str):
    train_data = pd.read_csv(train_data_path)
    path_lst = []
    for idx, row in train_data.iterrows():
        path_lst.append((row.img_path, row.A, row.B, row.C, row.D, row.E, row.F, row.G, row.H, row.I, row.J))
    return path_lst # (path, labels)

@st.experimental_memo
def get_checked_class_data(path_list:list, check_list: list):
    return_list = []
    for p in path_list:
        for idx, label in enumerate(p[1:]):
            if label == 1 and CLASSES[idx] in check_list:
                return_list.append(p)
    return return_list

data_list = set_data('./data/train.csv')

checked_label = st.multiselect(label='레이블 선택', options=CLASSES, default=CLASSES)
data_list = get_checked_class_data(data_list, checked_label)
# TODO
# 클래스 선택하면 그 클래스가 들어있는 사진 리스트 가져오기


path = st.selectbox(
    label='이미지 선택',
    options=data_list,
    format_func=lambda x: x[0][8:])

img_path = os.path.join(DATA_PATH, path[0])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

col1, col2, col3 = st.columns(3)
with col1:
    pass
with col2:
    st.image(img)
    tmp = []
    for idx, l in enumerate(path[1:]):
        if l == 1:
            tmp.append(CLASSES[idx])
    st.text('포함된 클래스: ' + ', '.join(tmp))
with col3:
    pass




