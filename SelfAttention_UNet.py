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

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict

from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

os.environ['TORCH_HOME'] = "/home/xyb/Lab/TorchModels"

CFG = {
    "seed": 1979,
    "arch": "UnetPlusPlus",
    "encoder_name": 'efficientnet-b7', ## efficientnet-b7 / timm-efficientnet-b8 / timm-efficientnet-l2 / timm-resnest200e 
    "encoder_weights": 'imagenet', ## noisy-student / imagenet
    "weight_decay": 1e-6,
    "LOSS": "bce_tversky",
    "train_bs": 66,
    "img_size": [224, 224], ## [224, 224]
    "epochs": 15,
    "lr": 2e-3, #2e-3,
    "optimizer": 'AdamW',
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6, #1e-6,
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if CFG['wandb']:
    import wandb
    try:
        wandb.login(key="67871c2e8f97fa74b52c18bcfccbee7fee0361d2")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project="UMWGI", 
                    #name="York_PPPM",
                    config=class2dict(CFG),
                    #group="DeBERTa-V3L",
                    #job_type="train",
                    )
import logging
logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. 💭Model:{CFG['arch']}---{CFG['encoder_name']} KFold={CFG['n_fold']} 🔴🟡🟢 {sys.argv}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(CFG['seed'])

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
            return image

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

def prepare_loaders(fold):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = CustomDataset(train_df, train_flag=True, load_mask=True, transforms=data_transforms['train'])
    valid_dataset = CustomDataset(valid_df, train_flag=True, load_mask=True, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], 
                              num_workers=32, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], 
                              num_workers=32, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

# ====================================================
# Reading Metadata
# ====================================================
def add_image_paths(df, channels=3, stride=2):
    for i in range(channels):
        df[f"image_path_{i:02}"] = df.groupby(["case", "day"])["image_path"].shift(-i * stride).fillna(method="ffill")
    
    image_path_columns = [f"image_path_{i:02d}" for i in range(channels)]
    df["image_paths"] = df[image_path_columns].values.tolist()
    df = df.drop(columns=image_path_columns)
    return df

BASE_PATH = "./ram"
df = pd.read_csv(f'{BASE_PATH}/mask/train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len) # length of each rle mask
df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy', regex=True).str.replace("/kaggle/input/uwmgi-mask-dataset", f"{BASE_PATH}/mask")
df['image_path'] = df.image_path.str.replace("/kaggle/input/uw-madison-gi-tract-image-segmentation", f"{BASE_PATH}")

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks

df = add_image_paths(df, channels=3, stride=2)

# ====================================================
# GroupKFold
# ====================================================
skf = StratifiedGroupKFold(n_splits=CFG['n_fold'], shuffle=True, random_state=CFG['seed'])
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
    df.loc[val_idx, 'fold'] = fold
#display(df.groupby(['fold','empty'])['id'].count())
   

# ====================================================
# Helpers
# ====================================================
def configure_optimizers(model, cfg):
    optimizer_kwargs = dict(
        params=model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
    )
    if cfg['optimizer'] == "Adadelta":
        optimizer = torch.optim.Adadelta(**optimizer_kwargs)
    elif cfg['optimizer'] == "Adagrad":
        optimizer = torch.optim.Adagrad(**optimizer_kwargs)
    elif cfg['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(**optimizer_kwargs)
    elif cfg['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(**optimizer_kwargs)
    elif cfg['optimizer'] == "Adamax":
        optimizer = torch.optim.Adamax(**optimizer_kwargs)
    elif cfg['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(**optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    return optimizer

def fetch_scheduler(optimizer):
    if CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['T_max'], 
                                                   eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG['T_0'], 
                                                             eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG['min_lr'],)
    elif CFG['scheduler'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG['scheduler'] == None:
        return None
        
    return scheduler

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

# ====================================================
# Model
# ====================================================
import segmentation_models_pytorch as smp
from AttentionUNet import AttentionUNet as attUNet
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
        """
        self.model = smp.create_model(
            self.cfg['arch'],
            encoder_name=self.cfg['encoder_name'],
            encoder_weights=self.cfg['encoder_weights'],
            in_channels=3,
            classes=3,
            activation=None,
        )
        """
        self.model = attUNet(num_classes=3)
        self.loss_fn = self._init_loss_fn()

    def _init_loss_fn(self):
        losses = self.cfg["LOSS"].split("_")
        loss_fns = [self.LOSS_FNS[loss] for loss in losses]

        def criterion(y_pred, y_true):
            return sum(loss_fn(y_pred, y_true) for loss_fn in loss_fns) / len(loss_fns)

        return criterion

    def forward(self, images):
        return self.model(images)


def train_one_epoch(model, optimizer, scheduler, dataloader, epoch, fold):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train[{fold}] ')
    for step, (images, masks) in pbar:         
        images = images.to(DEVICE, dtype=torch.float)
        masks  = masks.to(DEVICE, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=False):
            y_pred = model(images)
            loss   = model.loss_fn(y_pred, masks)
            loss   = loss / CFG['n_accumulate']
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        total_epoch = CFG['epochs']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB',
                        epoch=f"{epoch}/{total_epoch}",)
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, epoch, fold):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid[{fold}] ')
    for step, (images, masks) in pbar:        
        images  = images.to(DEVICE, dtype=torch.float)
        masks   = masks.to(DEVICE, dtype=torch.float)

        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = model.loss_fn(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        total_epoch = CFG['epochs']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB',
                        epoch=f'{epoch}/{total_epoch}')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training(model, optimizer, scheduler, fold, num_epochs):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    logger.info(f"Running fold--->{fold} 🌱🌱🌱⚡️🍄🍄🍄")
    train_loader, valid_loader = prepare_loaders(fold=fold)
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           epoch=epoch, fold=fold)
        
        val_loss, val_scores = valid_one_epoch(model, valid_loader,                                                 
                                                 epoch=epoch, fold=fold)
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        # Log the metrics
        if CFG['wandb']:
        	wandb.log({"Train Loss": train_loss, 
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR":scheduler.get_last_lr()[0]})
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        logger.info(f"LR::{scheduler.get_last_lr()[0]:.6f} 〽️")
        logger.info(f'Epoch:{epoch}/{num_epochs} ## Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            logger.info(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            if CFG['wandb']:
                wandb.summary["Best Dice"]    = best_dice
                wandb.summary["Best Jaccard"] = best_jaccard
                wandb.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")
            logger.info(f"Model Saved ---> {PATH} 💯🆕!!")
            
        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        logger.info('Epoch {} complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(epoch,
            epoch_time_elapsed // 3600, (epoch_time_elapsed % 3600) // 60, (epoch_time_elapsed % 3600) % 60))

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
        print();
    
    end = time.time()
    time_elapsed = end - start
    print('Training[{}] complete in {:.0f}h {:.0f}m {:.0f}s'.format(fold,
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: jaccard={:.4f}, dice={:.4f} in epoch[{}]".format(best_jaccard, best_dice, best_epoch))
    
    logger.info('Training[{}] complete in {:.0f}h {:.0f}m {:.0f}s ⛳️🐢🐢🐢'.format(fold,
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("Best Score: jaccard={:.4f}, dice={:.4f} in epoch[{}]🌟✨✨\n".format(best_jaccard, best_dice, best_epoch))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

SINGLE_FOLD = -1
for fold in range(CFG['n_fold']):    
    if (SINGLE_FOLD >= 0) and (fold != SINGLE_FOLD):
        print(f"☠️☠️☠️ Only run fold={SINGLE_FOLD}, fold[{fold}] skipped!!!🏮🏮🏮")
        logger.info(f"☠️☠️☠️ Only run fold={SINGLE_FOLD}, fold[{fold}] skipped!!!🏮🏮🏮")
        continue

    print(f'#'*15)
    print(f'### Fold: {fold}')
    print(f'#'*15)
   
    model = CustomModel(CFG).to(DEVICE)
    optimizer = configure_optimizers(model, CFG)  
    scheduler = fetch_scheduler(optimizer)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    model, history = run_training(model, optimizer, scheduler, 
                                  fold=fold, num_epochs=CFG['epochs'])
if CFG['wandb']:     
    wandb.finish()
    #display(ipd.IFrame(run.url, width=1000, height=720))





