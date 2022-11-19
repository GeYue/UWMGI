

import numpy as np
import pandas as pd
from glob import glob

#df = pd.read_csv('../input/uwmgi-mask-dataset/train.csv')
BASE_PATH = "./data"
df = pd.read_csv(f'{BASE_PATH}/mask/train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len) # length of each rle mask

#df['image_path'] = df.image_path.str.replace('/kaggle/','../') # .str: 特定列应用python字符串处理方法 
#df['mask_path'] = df.mask_path.str.replace('/kaggle/','../')

#df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')

df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy', regex=True).str.replace("/kaggle/input/uwmgi-mask-dataset", f"{BASE_PATH}/mask")
df['image_path'] = df.image_path.str.replace("/kaggle/input/uw-madison-gi-tract-image-segmentation", f"{BASE_PATH}")

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks
print(df.head())
print("=============================")

channels=5
stride=1
for i in range(channels):
    df[f'image_path_{i:02}'] = df.groupby(['case','day'])['image_path'].shift(-i*stride).fillna(method="ffill")
df['image_paths'] = df[[f'image_path_{i:02d}' for i in range(channels)]].values.tolist()
df.image_paths[0]
print(df.head())
print("=============================")

IMAGE_DIR = './data/generation/images'
ids = df['id'].unique()
print(ids)
print("=============================")

import cv2
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")
    return img

def load_imgs(img_paths):
    imgs = [load_img(path) for path in img_paths]
    imgs = np.stack(imgs, axis=-1)
    return imgs

def save_imgs(id_):
    row = df[df['id']==id_].squeeze()
    
    img_paths = row.image_paths
    imgs = load_imgs(img_paths)
    np.save(f'{IMAGE_DIR}/{id_}.npy', imgs)
    
    return


from joblib import Parallel, delayed
from tqdm import tqdm
_ = Parallel(n_jobs=-1, backend='threading')(delayed(save_imgs)(id_) for id_ in tqdm(ids, total=len(ids)))

