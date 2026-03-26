import os
import sys
import torch
import argparse
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Set distributed environment variables BEFORE importing chexfound modules
# This prevents "Can't initialize PyTorch distributed environment" error
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('LOCAL_WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')

# Setup paths for imports
script_dir = os.path.dirname(__file__)
code_root = os.path.join(script_dir, '..')
chexfound_root = os.path.join(code_root, 'model', 'chexfound')
chexfound_repo = os.path.join(chexfound_root, 'CheXFound')

sys.path.insert(0, chexfound_repo)  # For chexfound package
sys.path.insert(0, code_root)       # For dataset, loss, etc.

from scripts.dataset import ChestXrayDataset
from loss.loss_chexfound import WeightedFocalBCELoss, calculate_effective_weights
from model_initialization import (
    setup_foundation_model,
    setup_glori_head,
    resize_decoder_for_num_classes,
    prepare_model_for_training,
    get_glori_parameters,
)
from train_utils import train_one_epoch, validate, get_scheduler, get_model, ModelEMA

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################################

target_class = 'Cardiomegaly'
assert target_class in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

base_dir = os.path.join(chexfound_root, 'CheXFound_Model')  # Use relative path from chexfound root
os.makedirs(base_dir, exist_ok=True)

config_file = os.path.join(base_dir, 'config.yaml')
pretrained_weights = os.path.join(base_dir, 'teacher_checkpoint.pth')
classifier_fpath = os.path.join(base_dir, 'glori.pth')
classifier_json = os.path.join(base_dir, 'results_eval_linear.json')
output_dir = os.path.join(base_dir, 'example')
os.makedirs(output_dir, exist_ok=True)

#################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU (default: True)')

parser.set_defaults(
    config_file=config_file,  # path to architecture configuration files
    pretrained_weights=None,
    output_dir=output_dir,
    opts=[],
    image_size=512,
    patch_size=16,
    n_register_tokens=4,
    n_last_blocks=4,
    return_class_token=True,
    num_classes=40,
    num_heads=8,
)
args, unknown = parser.parse_known_args()

# Determine device based on args
USE_GPU = args.use_gpu and torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
print(f"Using device: {DEVICE} (GPU available: {torch.cuda.is_available()})")

#################################################################
# Setup models
#################################################################

# set up foundation model
base_model, autocast_dtype = setup_foundation_model(args, pretrained_weights, device=DEVICE)

# set up glori head
glori = setup_glori_head(args, base_model, classifier_json, classifier_fpath, device=DEVICE)

#################################################################
# Resize decoder for target number of classes
#################################################################

TARGET_CLASSES = 14
ORIG_CLASSES = 40

resize_decoder_for_num_classes(glori, TARGET_CLASSES, ORIG_CLASSES, device=DEVICE)

# Freeze backbone
for p in base_model.parameters():
    p.requires_grad = False

#################################################################

EPOCHS=3

# -----------------------------
# Run Training
# -----------------------------

# # Prepare for cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=32)
fold_results = []

df_all = pd.read_csv(os.path.join(code_root, 'data/grand-xray-slam-division-a-mini/train1-mini.csv'))
df_all = df_all[~df_all['Image_name'].isin([
    '00043046_001_001.jpg',
    '00052495_001_001.jpg',
    '00056890_001_001.jpg'
])]

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

# Define transforms
train_tfms = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, p=0.4),
    A.CLAHE(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.CoarseDropout(max_holes=1, max_height=20, max_width=20, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=0, p=0.25),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

    

best_auc = -1.0   # track best AUC
best_loss = float("inf")
# Create datasets
train_df = df_all.copy()
#val_df = df_all.iloc[val_idx].copy()
# Model is already on device

# -----------------------------
# Loss & Optimizer
# -----------------------------

#criterion = FocalLoss(alpha=0.25, gamma=2.0)
pos_weights = calculate_effective_weights(train_df, label_cols, beta=0.9999)

# Prepare model for training
model, device = prepare_model_for_training(base_model, glori, args, freeze_backbone=True, use_data_parallel=False, device=DEVICE)

# Setup loss with weighted sampling
pos_weights = calculate_effective_weights(train_df, label_cols, beta=0.9999)
criterion = WeightedFocalBCELoss(alpha=0.25, gamma=2.0, pos_weights=pos_weights.to(device))

# Get trainable parameters and setup optimizer
glori_trainable = get_glori_parameters(model)
print("Trainable glori params count:", sum(p.numel() for p in glori_trainable))

optimizer = torch.optim.AdamW(glori_trainable, lr=2e-4, weight_decay=1e-5)

# Remove distributed environment check by setting flags before any initialization
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Setup EMA
ema = ModelEMA(model, decay=0.99967)

    
# Dataset
train_dataset = ChestXrayDataset(
    df=train_df,
    img_dir=os.path.join(code_root, "data/grand-xray-slam-division-a-mini/train1-mini/output"),  # Use relative path from current script
    is_train=True,
    train_transform=train_tfms,
    val_transform=val_tfms
)

# # (for a real setup, you'd split train/val from train1.csv)
# val_dataset = ChestXrayDataset(
#     df=val_df,
#     img_dir="/kaggle/input/xray-data-div-b-600p-95/train2_resized",
#     transform=val_tfms
# )

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,   
    #num_workers=4,   
    pin_memory=True
)
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=64,
#     shuffle=False,
#     num_workers=16,
#     #num_workers=4,
#     pin_memory=True
# )

for epoch in range(EPOCHS):
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, ema)

    #if epoch >1:
            
    #val_loss, val_auc, per_label_auc = validate(ema.ema, val_loader, criterion)
    print(f"[Epoch {epoch+1}] TrainLoss={train_loss:.4f} ")




    save_path = f"model{epoch+1}.pth"
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": get_model(model).state_dict(),  
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema.state_dict()
    }, save_path)

    
    print(f"✅ Saved model  ")
    
#################################################################

save_path = f"best_model_f{epoch+1}.pth"
torch.save({
    "epoch": epoch+1,
    "model_state_dict": get_model(model).state_dict(),  
    "optimizer_state_dict": optimizer.state_dict()
}, save_path)

# also save EMA
if ema is not None:
    torch.save({
        "epoch": epoch+1,
        "ema_state_dict": ema.state_dict()
    }, f"ema_model_f{epoch+1}.pth")

print(f"✅ Saved model  ")