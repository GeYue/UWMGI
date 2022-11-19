#coding=UTF-8

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import random
from glob import glob
import os, sys, shutil, time, copy, gc, cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict

os.environ['TORCH_HOME'] = "/home/xyb/Lab/TorchModels"
CFG = {
    "seed": 1979,
    "arch": "Unet",
    "encoder_name": 'efficientnet-b4',
    "encoder_weights": 'imagenet',
    "weight_decay": 1e-6,
    "LOSS": "bce_tversky",
    "train_bs": 64,
    "img_size": [384, 384],
    "epochs": 15,
    "lr": 2e-3,
    "optimizer": 'AdamW',
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_0": 25,
    "warmup_epochs": 0,
    "n_accumulate": 1,
    "n_fold": 5,
    "num_classes": 3,
    "wandb": True,
    "full_loading": True,
    "2.5D": True,
}

CFG['valid_bs'] = CFG["train_bs"]*2
CFG['T_max'] = int(30000/CFG['train_bs']*CFG['epochs'])+50


def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
set_seed(CFG['seed'])

### Dataset for DataLoader
data_transforms = {
    "train": A.Compose([
        A.Resize(CFG['img_size'][0], CFG['img_size'][1], interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
##          A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=CFG['img_size'][0]//20, max_width=CFG['img_size'][1]//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ToTensorV2(transpose_mask=True),
        ], p=1.0),
    
    "valid": A.Compose([
        A.Resize(CFG['img_size'][0], CFG['img_size'][1], interpolation=cv2.INTER_NEAREST),
        ToTensorV2(transpose_mask=True),
        ], p=1.0)
}

def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
#     row['id'] = f'case{case}_day{day}_slice_{slice_}'     
    return row

def add_image_paths(df, channels=3, stride=2):
    for i in range(channels):
        df[f"image_path_{i:02}"] = df.groupby(["case", "day"])["image_path"].shift(-i * stride).fillna(method="ffill")
    
    image_path_columns = [f"image_path_{i:02d}" for i in range(channels)]
    df["image_paths"] = df[image_path_columns].values.tolist()
    df = df.drop(columns=image_path_columns)
    return df

class CustomDataset(Dataset):    
    def __init__(self, df: pd.DataFrame, train_flag: bool, load_mask: bool, transforms: Optional[Callable] = None):
        self.df = df
        self.isTrain = train_flag
        self.using25D = CFG['2.5D']
        self.load_mask = load_mask
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.using25D:
            image = self._load_images(row["image_paths"])
        else:
            image = self._load_image(row["image_path"])

        if self.load_mask:
            mask = self._load_mask(row["mask_path"])
            if self.transforms:
                data = self.transforms(image=image, mask=mask)
                image, mask = data["image"], data["mask"]

            return image, mask
        else:
            id_ = row["id"]
            h, w = image.shape[:2]

            if self.transforms:
                data = self.transforms(image=image)
                image = data["image"]

            return image, id_, h, w

    def _load_images(self, paths):
        images = [self._load_image(path, tile=False) for path in paths]
        image = np.stack(images, axis=-1)
        return image

    @staticmethod
    def _load_image(path, tile: bool = True):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype("float32")  # original is uint16
        
        if tile:
            image = np.tile(image[..., None], [1, 1, 3])  # gray to rgb
            
        image /= image.max()

        return image

    @staticmethod
    def _load_mask(path):
        return np.load(path).astype("float32") / 255.0

import segmentation_models_pytorch as smp
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LOSS_FNS = {
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "dice": smp.losses.DiceLoss(mode="multilabel"),
        "focal": smp.losses.FocalLoss(mode="multilabel"),
        "jaccard": smp.losses.JaccardLoss(mode="multilabel"),
        "lovasz": smp.losses.LovaszLoss(mode="multilabel"),
        "tversky": smp.losses.TverskyLoss(mode="multilabel"),}

        self.cfg = config
        self.model = smp.create_model(
            self.cfg['arch'],
            encoder_name=self.cfg['encoder_name'],
            encoder_weights=self.cfg['encoder_weights'],
            in_channels=5,
            classes=3,
            activation=None,
        )
        self.loss_fn = self._init_loss_fn()

    def _init_loss_fn(self):
        losses = self.cfg["LOSS"].split("_")
        loss_fns = [self.LOSS_FNS[loss] for loss in losses]

        def criterion(y_pred, y_true):
            return sum(loss_fn(y_pred, y_true) for loss_fn in loss_fns) / len(loss_fns)

        return criterion

    def forward(self, images):
        return self.model(images)

import cupy as cp
def mask2rle(mask):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    mask = cp.array(mask)
    pixels = mask.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def masks2rles(masks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx in range(masks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        mask = cv2.resize(masks[idx], dsize=(width, height), interpolation=cv2.INTER_NEAREST)  # back to original shape

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(mask[..., midx])

        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])

    return pred_strings, pred_ids, pred_classes


tqdm.pandas()
sub_df = pd.read_csv('./data/sample_submission.csv')
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv('./data/train.csv')[:1000*3]
    sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
sub_df = sub_df.progress_apply(get_metadata,axis=1)

if debug:
    paths = glob(f'./data/train/**/*png',recursive=True)
#     paths = sorted(paths)
else:
    paths = glob(f'/kaggle/input/uw-madison-gi-tract-image-segmentation/test/**/*png',recursive=True)
#     paths = sorted(paths)

path_df = pd.DataFrame(paths, columns=['image_path'])
cache_file = Path("./inference_cach.csv")
cached = cache_file.exists()
if cached:
    path_df = pd.read_csv('./inference_cach.csv')
else:
    path_df = path_df.progress_apply(path2info, axis=1)
    path_df.to_csv('./inference_cach.csv')
path_df.head()

test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')
test_df = add_image_paths(test_df, channels=5, stride=1)
test_df.head()


test_dataset = CustomDataset(test_df, train_flag=False, load_mask=False, transforms=data_transforms['valid'])
test_loader = DataLoader(test_dataset, batch_size=CFG['valid_bs'],
                         num_workers=2, shuffle=False, pin_memory=True)


@torch.no_grad()
def Inference(model, dataloader, model_paths):
    model.eval()
    bFirstPath = True
    tt_outputs = None; tt_tail_outputs = None;
    out_ids = None; out_h = None; out_w = None; tail_ids = None; tail_h = None; tailw3 = None;
    for path in model_paths:
        model.load_state_dict(torch.load(f'{path}', map_location = DEVICE))
        outputs = None; tail_outputs = None;
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Inference[{CFG['encoder_name']}]")
        for step, (images, ids, heights, widths) in pbar:        
            images = images.to(DEVICE, dtype=torch.float)
            size = images.size()
            out = model(images)
            out = nn.Sigmoid()(out)
            if size[0] == CFG['valid_bs']:
                outputs = np.append(outputs, np.expand_dims(out.cpu().detach().numpy(), axis=0), axis=0) if outputs is not None else np.expand_dims(out.cpu().detach().numpy(), axis=0)
                if not bFirstPath:
                    continue
                out_ids = np.append(out_ids, np.expand_dims(ids, axis=0), axis=0) if out_ids is not None else np.expand_dims(ids, axis=0)
                out_h = np.append(out_h, np.expand_dims(heights, axis=0), axis=0) if out_h is not None else np.expand_dims(heights, axis=0)
                out_w = np.append(out_w, np.expand_dims(widths, axis=0), axis=0) if out_w is not None else np.expand_dims(widths, axis=0)
            else:
                tail_outputs = np.expand_dims(out.cpu().detach().numpy(), axis=0)
                if not bFirstPath:
                    continue
                tail_ids = np.expand_dims(ids, axis=0)
                tail_h = np.expand_dims(heights, axis=0)
                tail_w = np.expand_dims(widths, axis=0)
        bFirstPath = False
        tt_outputs = tt_outputs + outputs if tt_outputs is not None else outputs
        tt_tail_outputs = tt_tail_outputs + tail_outputs if tt_tail_outputs is not None else tail_outputs

    del outputs, tail_outputs
    gc.collect()

    tt_outputs =  tt_outputs / len(model_paths)
    tt_tail_outputs = tt_tail_outputs / len(model_paths)

    return tt_outputs, out_ids, out_h, out_w, tt_tail_outputs, tail_ids, tail_h, tail_w

def EnsembleModels(allModel_rtnValue, threshold):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    (tt_outputs, out_ids, out_h, out_w, tt_tail_outputs, tail_ids, tail_h, tail_w) = allModel_rtnValue
    for idx in range(len(tt_outputs)):
        zeros = np.zeros((tt_outputs.shape[1], 3, CFG['img_size'][0], CFG['img_size'][1]))
        zeros = np.transpose(zeros, (0, 2, 3, 1))
        msk = (np.transpose(tt_outputs[idx], (0, 2, 3, 1))) > threshold
        msk = msk + zeros
        result = masks2rles(msk, out_ids[idx], out_h[idx], out_w[idx])

        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

    for idx in range(len(tt_tail_outputs)):
        zeros = np.zeros((tt_tail_outputs.shape[1], 3, CFG['img_size'][0], CFG['img_size'][1]))
        zeros = np.transpose(zeros, (0, 2, 3, 1))
        msk = (np.transpose(tt_tail_outputs[idx], (0, 2, 3, 1))) > threshold
        msk = msk + zeros
        result = masks2rles(msk, tail_ids[idx], tail_h[idx], tail_w[idx])

        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

    pred_df = pd.DataFrame({"id": pred_ids, "class": pred_classes, "predicted": pred_strings})
    return pred_df


DEVICE = "cuda"
model = CustomModel(CFG).to(DEVICE)
model_paths  = glob(f'/home/xyb/Lab/Kaggle/UWMGI/pretrained/eff7-img352/best_epoch*.bin')
tt_outputs1, out_ids, out_h, out_w, tt_tail_outputs1, tail_ids, tail_h, tail_w = Inference(model, test_loader, model_paths)
del out_ids, out_h, out_w, tail_ids, tail_h, tail_w
gc.collect()

print (f"")
CFG["encoder_name"] = 'timm-efficientnet-l2'
CFG["encoder_weights"] = 'noisy-student'
model = CustomModel(CFG).to(DEVICE)
model_paths  = glob(f'/home/xyb/Lab/Kaggle/UWMGI/pretrained/tEffL2/best_epoch*.bin')
tt_outputs2, out_ids, out_h, out_w, tt_tail_outputs2, tail_ids, tail_h, tail_w = Inference(model, test_loader, model_paths)

allModel_rtnValue = ((tt_outputs1+tt_outputs2)/2, out_ids, out_h, out_w, (tt_tail_outputs1+tt_tail_outputs2)/2, tail_ids, tail_h, tail_w)
pred_df = EnsembleModels(allModel_rtnValue, threshold=0.45)
del allModel_rtnValue

if not debug:
    sub_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
    del sub_df["predicted"]
else:
    sub_df = pd.read_csv("./data/train.csv")[: 1000 * 3]
    del sub_df["segmentation"]

sub_df = sub_df.merge(pred_df, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)
sub_df.head(5)


