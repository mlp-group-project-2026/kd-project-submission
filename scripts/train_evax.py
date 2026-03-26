# 
# !wget https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_base_patch16_merged520k_mim.pt

# 
import torch
from timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed, resample_patch_embed
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score 
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.io as io
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from torch.amp import autocast, GradScaler
from torchvision.transforms import autoaugment, transforms
from torchvision.io import read_image
from torchvision import transforms as T
# Use the recommended v2 transforms
from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup
import copy

# 
torch.manual_seed(48)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from scripts.dataset import ChestXrayDataset
from scripts.train_utils import ModelEMA, get_model, get_scheduler, train_one_epoch, checkpoint_filter_fn

# 
img_size = 448  


class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



class EVAX_Model(nn.Module):
    def __init__(self, unfreeze_last_n_blocks=4, pretrained_path=None):
        super().__init__()
        # Default path - can be overridden
        if pretrained_path is None:
            pretrained_path = '../model/evax/eva_x_base_patch16_merged520k_mim.pt'
        self.backbone = self._build_backbone(pretrained_path=pretrained_path)
        
        self.backbone.head=nn.Linear(in_features=768, out_features=14, bias=True)

        # # Also unfreeze head and normalisation layers around it
        for p in self.backbone.head.parameters():
            p.requires_grad = True

    
    
    def forward(self, x):
        backbone_output = self.backbone(x)
    
        #return self.head(backbone_output)
        return backbone_output

    def _build_backbone(self, pretrained_path):
        model = EVA_X(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            qkv_fused=False,
            mlp_ratio=4 * 2 / 3,
            swiglu_mlp=True,
            scale_mlp=True,
            use_rot_pos_emb=True,
            ref_feat_shape=(14, 14),  # 224/16
        )
        eva_ckpt = checkpoint_filter_fn(
            torch.load(pretrained_path, map_location='cpu', weights_only=False),
            model,
        )
        msg = model.load_state_dict(eva_ckpt, strict=False)
        print(msg)
        return model

# 
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# Determine device and create scaler appropriately
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    scaler = GradScaler(device="cuda") if DEVICE.type == "cuda" else GradScaler()
except TypeError:
    # Fallback for older PyTorch versions
    scaler = GradScaler()

# Using consolidated `train_one_epoch` from scripts.train_utils



# -----------------------------
# Validation (macro + per-label AUROC)
# -----------------------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_outputs = [], []

    progress_bar = tqdm(loader, desc="[Val]")
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device).float()
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.sigmoid().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)

    per_label_auc = {}
    for i, label in enumerate(loader.dataset.label_cols):
        try:
            per_label_auc[label] = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            per_label_auc[label] = np.nan

    try:
        #macro_auc = roc_auc_score(all_labels, all_outputs, average="macro", multi_class="ovo")
        macro_auc = roc_auc_score(all_labels, all_outputs, average="macro")
    except ValueError:
        macro_auc = np.nan

    return running_loss / len(loader.dataset), macro_auc, per_label_auc




# 
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if target=1, 1-p otherwise
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.alpha = 0.7  # Weight between focal and BCE
        
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * bce_loss

# Calculate better class weights
def calculate_effective_weights(df, label_cols, beta=0.999):
    """Calculate more robust class weights using effective number of samples"""
    n = len(df)
    weights = []
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = n - pos_count
        
        # Effective number of samples (from Class-Balanced Loss paper)
        eff_pos = (1 - beta**pos_count) / (1 - beta) if pos_count > 0 else 0
        eff_neg = (1 - beta**neg_count) / (1 - beta) if neg_count > 0 else 0
        
        weight = eff_neg / (eff_pos + 1e-8)  # Avoid division by zero
        weights.append(min(weight, 10.0))  # Cap extreme weights
    
    return torch.tensor(weights).float()



# 
def get_model(module):
    """Return the underlying model (unwrap DataParallel / DDP)."""
    return module.module if hasattr(module, "module") else module

# 
#from sklearn.model_selection import GroupKFold

EPOCHS=7

# -----------------------------
# Run Training
# -----------------------------

# Prepare for cross-validation
#kf = KFold(n_splits=5, shuffle=True, random_state=32)
#gkf = GroupKFold(n_splits=5)

# Data paths - make configurable
DATA_CSV_PATH = './kd-project/data/grand-xray-slam-division-a-mini/train1-mini.csv'  # Update this path
DATA_IMG_DIR = './kd-project/data/grand-xray-slam-division-a-mini/train1-mini/output'  # Update this path

fold_results = []

# Load data - with graceful fallback for testing
if os.path.exists(DATA_CSV_PATH):
    df_all = pd.read_csv(DATA_CSV_PATH)
    df_all = df_all[~df_all['Image_name'].isin([
        '00043046_001_001.jpg',
        '00052495_001_001.jpg',
        '00056890_001_001.jpg'
    ])]
else:
    print(f"Warning: CSV not found at {DATA_CSV_PATH}. Please update DATA_CSV_PATH.")
    df_all = pd.DataFrame()  # Empty dataframe for testing

label_cols = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]

#for fold, (train_idx, val_idx) in enumerate(kf.split(df_all)):
#for fold, (train_idx, val_idx) in enumerate(gkf.split(df_all, groups=df_all['Patient_ID'])):
    
#print(f"\n=== Training Fold {fold+1}/5 ===")
    
best_auc = -1.0   # track best AUC
best_loss = float("inf")
# Create datasets
train_df = df_all.copy()
#val_df = df_all.iloc[val_idx].copy()

#print(f"Fold {fold}: Train {train_df['Patient_ID'].nunique()} patients, Val {val_df['Patient_ID'].nunique()} patients")

# Define transforms
train_tfms = v2.Compose([
    v2.ToImage(),
    v2.Resize((img_size, img_size)),
    v2.ToDtype(torch.float32, scale=True),
])

val_tfms = v2.Compose([
    v2.ToImage(),
    v2.Resize((img_size, img_size)),
    v2.ToDtype(torch.float32, scale=True),
])

# Device setup
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"CUDA available: {torch.cuda.is_available()}")

# Model: EVAX base
model = EVAX_Model(unfreeze_last_n_blocks=6, pretrained_path=None)

# -----------------------------
# Loss & Optimizer
# -----------------------------
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = FocalLoss(alpha=0.25, gamma=2.0)
pos_weights = calculate_effective_weights(train_df, label_cols, beta=0.9999)
criterion = WeightedFocalBCELoss(alpha=0.25, gamma=2.0, pos_weights=pos_weights.to(DEVICE))


optimizer = torch.optim.AdamW(
    [
        {'params': model.backbone.patch_embed.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6},  
        {'params': model.backbone.rope.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6},  
        {'params': model.backbone.blocks[0:4].parameters(), 'lr': 1e-6, 'weight_decay': 1e-4},  
        {'params': model.backbone.blocks[4:6].parameters(), 'lr': 5e-6, 'weight_decay': 1e-4},  
        {'params': model.backbone.blocks[6:8].parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  
        {'params': model.backbone.blocks[8:10].parameters(), 'lr': 3e-5, 'weight_decay': 1e-4},  
        {'params': model.backbone.blocks[10:].parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},  
        {'params': model.backbone.fc_norm.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},  
        {'params': model.backbone.head.parameters(), 'lr': 8e-5, 'weight_decay': 0},      
    ]
)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"⚡ Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)  # wrap for multi-GPU

model = model.to(DEVICE)

#EMA
ema = ModelEMA(model, decay=0.9997)

# Dataset
train_dataset = ChestXrayDataset(
    df=train_df,
    img_dir=DATA_IMG_DIR,
    transform=train_tfms
)

# # (for a real setup, you'd split train/val from train1.csv)
# val_dataset = ChestXrayDataset(
#     df=val_df,
#     img_dir="/kaggle/input/600-p-div-a-data/train1_resized",
#     transform=val_tfms
# )

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,   
    #num_workers=4,   
    pin_memory=True
)
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=64,
#     shuffle=False,
#     num_workers=4,
#     #num_workers=4,
#     pin_memory=True
# )

for epoch in range(EPOCHS):
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device=DEVICE, ema=ema, accum_steps=2, scaler=scaler)
    #val_loss, val_auc, per_label_auc = validate(ema.ema, val_loader, criterion)

    print(f"[Epoch {epoch+1}] TrainLoss={train_loss:.4f} ")
    # print(f"[Epoch {epoch+1}] TrainLoss={train_loss:.4f} | EMA ValLoss={val_loss:.4f} | AUC={val_auc:.4f}")
    # print("Per-label AUROC:", {k: f"{v:.3f}" for k, v in per_label_auc.items()})

    if epoch > 3:
        #best_auc = val_auc
        save_path = f"best_model_f{epoch+1}.pth"
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": get_model(model).state_dict(),   # unwrap
            "optimizer_state_dict": optimizer.state_dict(),
        }, save_path)
    
        # also save EMA
        if ema is not None:
            torch.save({
                "epoch": epoch+1,
                "ema_state_dict": ema.state_dict(),
            }, f"ema_model_f{epoch+1}.pth")
    
        print(f"✅ Saved new best model")

# 



