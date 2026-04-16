import pandas as pd
import pydicom
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2

image_path = "./data/processed/INBREAST/AllDICOMs"
save_dir = "./data/processed/INBREAST/images/"

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv('./data/processed/INBREAST/metadata_old.csv', sep=';')

df['img_id'] = df['File Name'].astype(str) + '.png'
df['split'] = 'val'

def birads_to_label(birads):
    birads = str(birads).strip().lower()
    if birads in ['0', '1', '2']:
        return 'normal'
    elif birads == '3':
        return 'benign'
    elif birads in ['4', '4a', '4b', '4c', '5', '6']:
        return 'malignant'
    else:
        return 'unknown'
    
df['label'] = df['Bi-Rads'].apply(birads_to_label)

df.to_csv('./data/processed/INBREAST/metadata.csv')


images = [f for f in listdir(image_path) if isfile(join(image_path, f))]

for img in images:
    filename = img.split('_')[0]
    
    row = df.loc[df['File Name'] == filename]
    
    ds = pydicom.dcmread(image_path + '/' + img)
    img_array = ds.pixel_array

    img_normalized = cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_dir + filename + '.png', img_bgr)

