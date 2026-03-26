'''
=============================================================================
INFERENCE SCRIPT - CHEST X-RAY CLASSIFICATION
=============================================================================
Runs inference using CheXFound and EVA-X models, then ensembles predictions
'''

'''
SECTION 1: IMPORTS AND INITIAL SETUP
Load necessary libraries and test data
'''
import sys, os
from pathlib import Path

# Must be set before importing torch so MPS-unsupported ops fall back to CPU
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Add parent directory to path to enable imports from project root
sys.path.append(str(Path(__file__).parent.parent))



import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

import yaml
import argparse
import wandb


from dataset import ChestXrayDataset
from train_utils import ModelEMA
from model.evax.model import EVAX_Model
from scripts.model_initialization import CheXFoundWithGLoRIHead
from scripts.train_utils import get_model

import json
import torch
import argparse
import pandas as pd
from PIL import Image
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from chexfound.eval.utils import extract_hyperparameters_from_model
from chexfound.eval.classification.utils import setup_glori
from fvcore.common.checkpoint import Checkpointer
from matplotlib.colors import ListedColormap

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
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
from transformers import get_cosine_schedule_with_warmup
import copy
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from scripts.train_utils import checkpoint_filter_fn

from torch.utils.data.dataloader import default_collate

def safe_collate(batch):
    """Filter out None samples (missing images)"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0), []
    imgs, ids = zip(*batch)
    return default_collate(imgs), list(ids)


# 
'''
SECTION 2: CONFIGURATION
Define file paths and medical condition labels
'''

VALID_MACHINES = ["np", "avk", "tbsk", "eidf"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation.")
    parser.add_argument(
        "-m", "--machine",
        type=str,
        required=True,
        choices=VALID_MACHINES,
        help=f"Machine environment to use. Must be one of: {', '.join(VALID_MACHINES)}",
    )
    parser.add_argument(
        "-y", "--yaml",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-d", "--use_dataset",
        type=str,
        required=True,
        help="Type of dataset to use for inference. Must be one of: train/val/test",
    )
    parser.add_argument(
        "-t", "--teacher",
        type=str,
        default=None,
        help="Teacher logits shortcut name (defined in config.yaml teachers section). "
             "Omit to train without teacher logits.",
    )
    return parser.parse_args()


def load_config(machine: str, config_path: str) -> tuple[dict, dict, dict]:
    """Load full config and return (machine_cfg, training_cfg, teachers_cfg)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if machine not in config["machines"]:
        raise ValueError(f"Machine '{machine}' not found in {config_path}")

    machine_cfg = config["machines"][machine]
    training_cfg = config.get("training", {})
    teachers_cfg = config.get("teachers", {})
    return machine_cfg, training_cfg, teachers_cfg


print("Checking for GPU availability...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("Cuda not available.")


def main():
    args = parse_args()
    config_file = Path(__file__).parent.parent / args.yaml
    print(f"Loading config from: {config_file}")
    machine_cfg, train_cfg, teachers_cfg = load_config(args.machine, config_file)

    LOCAL = machine_cfg.get("run_locally", False)

    if LOCAL:
        print("⚡ Running in LOCAL mode with limited data for quick testing.")

    print(f"\nUsing machine config: '{args.machine}'")
    print(f"  data_path: {machine_cfg['data_path']}")

    use_dataset = args.use_dataset.lower()

    if use_dataset == 'train':
        print("Using TRAIN dataset for inference.")
        test_df_path = machine_cfg.get("inference_train_df", pd.DataFrame())  # Load test_df from config or default to empty DataFrame
        test_images_dir = machine_cfg.get("inference_train_images_dir", "")
    elif use_dataset == 'val':
        print("Using VALIDATION dataset for inference.")
        test_df_path = machine_cfg.get("inference_val_df", pd.DataFrame())
        test_images_dir = machine_cfg.get("inference_val_images_dir", "")
    elif use_dataset == 'test':
        print("Using TEST dataset for inference.")
        test_df_path = machine_cfg.get("inference_test_df", pd.DataFrame())
        test_images_dir = machine_cfg.get("inference_test_images_dir", "")
    else:
        print(f"Invalid dataset specified in config: {use_dataset}. Defaulting to TEST dataset.")
        use_dataset = 'test'


    test_df = pd.read_csv(test_df_path)


    chexfound_folder_path =machine_cfg.get("chexfound_folder_path", "")

    evax448_folder_path =machine_cfg.get("evax448_folder_path", "")

    output_folder_path = os.path.join(machine_cfg.get("save_dir", ""), f"full_{use_dataset}_set")
    os.makedirs(output_folder_path, exist_ok=True)

    chosen_num_workers = machine_cfg.get("num_workers", 4)

    run = wandb.init(
        project="mlp-project",
        name=f"{args.machine}_inference_ensemble",
        config={
            "machine": args.machine,
            "test_images_dir": machine_cfg.get("test_images_dir", ""),
            "chexfound_folder_path": machine_cfg.get("chexfound_folder_path", ""),
            "evax448_folder_path": machine_cfg.get("evax448_folder_path", ""),
            "output_folder_path": output_folder_path,
            "chexfound_batch_size": 64,
            "evax_batch_size": 128,
            "chexfound_img_size": 512,
            "evax_img_size": 448,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")),
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "num_workers": chosen_num_workers,
            "test_dataset_used": use_dataset,
            "test_df_path": test_df_path,
            "test_images_dir_path": test_images_dir,
        }
    )


    # Define 14 medical conditions to predict
    labels_cols=[
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


    '''
    =============================================================================
    MODEL 1: CHEXFOUND SETUP
    =============================================================================
    '''

    '''
    SECTION 3: CheXFound Environment Setup
    Set environment variables to prevent distributed training errors
    '''


    if LOCAL:
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        os.environ.setdefault('LOCAL_WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29008')

    # Add CheXFound repository to Python path
    repo_path = f"{chexfound_folder_path}/CheXFound"
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)



    # Set device (GPU if available)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    '''
    SECTION 5: CheXFound Model Configuration
    Set up paths and arguments for model loading
    '''

    base_dir = f'{chexfound_folder_path}/CheXFound_Model/'

    chexfound_config_file = base_dir + 'config.yaml'
    pretrained_weights = base_dir + 'chexfound_CheXFound_Model_teacher_checkpoint.pth'
    classifier_fpath = base_dir + 'chexfound_CheXFound_Model_glori.pth'
    classifier_json = base_dir + 'results_eval_linear.json'
    output_dir = base_dir + 'example'
    os.makedirs(output_dir, exist_ok=True)

    parser = argparse.ArgumentParser()

    parser.set_defaults(
        config_file=chexfound_config_file,  # path to architecture configuration files
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
    # Point output_dir to a temp location so CheXFound's write_config
    # doesn't overwrite our project config.yaml
    import tempfile
    args.output_dir = tempfile.mkdtemp(prefix="chexfound_")

    '''
    SECTION 6: Load CheXFound Backbone
    Load the feature extraction model (backbone)
    '''

    base_model, autocast_dtype = setup_and_build_model(args)

    '''
    SECTION 7: Setup GLoRI Classifier Head
    GLoRI sits on top of the backbone and outputs predictions (originally 40 classes)
    '''

    log_json = classifier_json
    with open(log_json, 'r') as f:
        content = f.read().split('\n')[-3]
        data = json.loads(content)
    best_classifier_str = data['best_classifier']['name']
    hyperparameters = extract_hyperparameters_from_model(best_classifier_str)
    learning_rate, avgpool, block = hyperparameters["lr"], hyperparameters["avgpool"], hyperparameters["blocks"]

    sample_input = torch.randn(1, 3, 512, 512).to(DEVICE)
    base_model = base_model.to(DEVICE)
    with torch.no_grad():
        sample_output = base_model.get_intermediate_layers(sample_input, n=args.n_last_blocks, return_class_token=True)

    glori, _ = setup_glori(
        sample_output=sample_output,
        n_last_blocks_list=block,
        learning_rates=learning_rate,
        avgpools=avgpool,
        num_classes=args.num_classes,
        multiview=False,
        decoder_dim=768,
        cat_cls=False,
    )


    '''
    SECTION 8: Resize GLoRI Head from 40 → 14 Classes
    Modify internal parameters to match our 14 medical conditions
    '''

    import torch.nn as nn

    TARGET_CLASSES = 14  # Our task
    ORIG_CLASSES = 40    # Original model

    # get classifier same way as before
    glori_wrapped = glori
    glori_module = getattr(glori_wrapped, "module", glori_wrapped)
    classifier_key = next(iter(glori_module.classifiers_dict.keys()))
    classifier = glori_module.classifiers_dict[classifier_key]
    decoder = getattr(classifier, "decoder", None)
    if decoder is None:
        raise RuntimeError("Decoder not found in classifier; cannot resize num_classes. Print classifier to inspect.")

    print("Decoder found:", type(decoder))
    print("Decoder.num_classes (before):", getattr(decoder, "num_classes", None))

    # 1) set decoder.num_classes
    decoder.num_classes = TARGET_CLASSES
    print("Set decoder.num_classes ->", decoder.num_classes)

    # 2) replace duplicate_pooling_bias if exists (common in this implementation)
    if hasattr(decoder, "duplicate_pooling_bias"):
        old = decoder.duplicate_pooling_bias
        print("Old duplicate_pooling_bias shape:", tuple(old.shape) if isinstance(old, torch.Tensor) else type(old))
        # create new bias param
        new_bias = nn.Parameter(torch.zeros(TARGET_CLASSES, dtype=torch.float32, device=DEVICE))
        # init small (optional)
        nn.init.constant_(new_bias, 0.0)
        # assign (if module originally on CPU, move)
        decoder.duplicate_pooling_bias = new_bias
        print("Replaced duplicate_pooling_bias with shape:", decoder.duplicate_pooling_bias.shape)
    else:
        print("No attribute duplicate_pooling_bias found on decoder — fine.")

    # 3) scan decoder for any 1D parameters/buffers whose length equals ORIG_CLASSES and replace them (safe heuristic)
    replaced = []
    for name, param in list(decoder.named_parameters()):
        if param.dim() == 1 and param.shape[0] == ORIG_CLASSES:
            print(f"Resizing param: {name} shape {param.shape} -> {TARGET_CLASSES}")
            parent = decoder
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            new_p = nn.Parameter(torch.zeros(TARGET_CLASSES, dtype=param.dtype, device=DEVICE))
            setattr(parent, attr, new_p)
            replaced.append(name)

    # also check buffers (non-parameter tensors)
    for name, buf in list(decoder.named_buffers()):
        if buf is None:
            continue
        if buf.ndim == 1 and buf.shape[0] == ORIG_CLASSES:
            print(f"Resizing buffer: {name} shape {buf.shape} -> {TARGET_CLASSES}")
            parent = decoder
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            # assign as buffer
            decoder.register_buffer(attr, torch.zeros(TARGET_CLASSES, dtype=buf.dtype, device=DEVICE))
            replaced.append("buffer:"+name)

    print("Replaced parameter/buffer names:", replaced)

    # 4) move whole glori to device again and do a quipretrained_weightsck forward test
    glori_wrapped.to(DEVICE)
    print("Moved glori to device.")

    # 
    '''
    SECTION 9: Define Complete CheXFound Model
    Combine backbone (feature extractor) + GLoRI head (classifier)
    '''

    # 
    '''
    SECTION 10: Helper Functions
    Utility functions for model unwrapping and data loading
    '''


    # 
    '''
    SECTION 11: CheXFound Prediction Function
    Loads checkpoint and runs inference on dataset
    '''

    scaler = GradScaler(device=DEVICE)

    def predict_chexfound(dataset,base_model,model_path='', batch_size=128, device="cuda"):
        fold_probs = []
        all_ids = None
        logits_list = dict()
        
        
        with torch.no_grad():
            
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0 if not torch.cuda.is_available() else chosen_num_workers, pin_memory=True, collate_fn=safe_collate)
            
            # Prepare model for multi-GPU if available
            print("Preparing model on device:", DEVICE, "GPUs:", torch.cuda.device_count())
            
            # 1) If glori is a DDP wrapper, unwrap to get the real module
            glori_wrapped = glori  # original object you loaded
            if hasattr(glori_wrapped, "module"):
                print("Unwrapping glori from DDP wrapper -> using glori.module")
                glori_clean = glori_wrapped.module
            else:
                glori_clean = glori_wrapped
            
            glori_clean = glori_clean.cpu()
            
            # Build complete model (backbone + classifier)
            model = CheXFoundWithGLoRIHead(backbone=base_model, glori_module=glori_clean, args=args)
            
            # Load trained weights from checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=DEVICE.type,
                weights_only=False  
            )
            model.load_state_dict(checkpoint["ema_state_dict"])

            # Move to GPU and wrap for multi-GPU if available
            model = model.to(DEVICE)
            if torch.cuda.device_count() > 1:
                print("Wrapping train_model with nn.DataParallel for multi-GPU")
                model = nn.DataParallel(model)
            
            model.eval()
            
            # Run inference
            fold_out = []
            ids = []

            with torch.no_grad():
                for images, img_id in tqdm(loader, desc="CheXFound Inference"):
                    images = images.to(DEVICE)
                    with autocast(device_type=DEVICE.type):
                        logits = model(images)  # Raw predictions
                    for i, sid in enumerate(img_id):
                        logits_list[sid] = logits[i].cpu().float().numpy()
                    probs = torch.sigmoid(logits)  # Convert to probabilities [0-1]
                    fold_out.append(probs.to(DEVICE))
                    if all_ids is None:
                        ids.extend(img_id)

            # Create dataframe from dictionary of logits list and save as CSV for analysis
            logits_df = pd.DataFrame.from_dict(logits_list, orient='index')
            # If we have the label names in scope, set them as columns
            if 'labels_cols' in locals() and len(logits_df.columns) == len(labels_cols):
                logits_df.columns = labels_cols
            # Make the index an explicit column called Image_name and save without the index
            logits_df.index.name = "Image_name"
            logits_df = logits_df.reset_index()
            logits_csv_path = f'{output_folder_path}/{model_path.split("/")[-1].split(".")[0]}_fullset_chexfound_logits_list.csv'
            logits_df.to_csv(logits_csv_path, index=False)
            artifact = wandb.Artifact(name=f"chexfound_{model_path.split('/')[-1].split('.')[0]}_fullset_chexfound_logits", type="predictions")
            artifact.add_file(logits_csv_path)
            run.log_artifact(artifact)

            fold_out = torch.cat(fold_out, dim=0)  # [N, C]
            fold_probs.append(fold_out)

            if all_ids is None:
                all_ids = ids

            final_probs = torch.stack(fold_probs, dim=0).mean(0).cpu()  # [N, C]
            return all_ids, final_probs


    # 
    '''
    SECTION 12: Define Test Transforms for CheXFound
    Resize to 512x512, normalize, convert to tensor (NO augmentation)
    '''

    chexfound_img_size = 512
    size_tuple = (chexfound_img_size, chexfound_img_size)   # IMPORTANT: albumentations expects a tuple

    test_tfms = A.Compose([
        A.Resize(height=chexfound_img_size, width=chexfound_img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2(),
    ])


    # 
    '''
    SECTION 13: Run CheXFound Inference - Fold 5
    '''

    test_dataset = ChestXrayDataset(
        df=test_df,
        img_dir=test_images_dir,
        dataset_type="test",
        transform=test_tfms
    )

    all_ids,final_probs = predict_chexfound(
        test_dataset,
        base_model,
        model_path=f'{chexfound_folder_path}/CheXFound/ema_model_f5.pth',
        batch_size=64,
        device=DEVICE
    )

    all_preds = np.vstack(final_probs)

    # Build submission dataframe
    sample_df = test_df

    submission_chexfound_ep5 = pd.DataFrame(all_preds, columns=labels_cols)
    submission_chexfound_ep5.insert(0, "Image_name", all_ids)
    submission_chexfound_ep5 = submission_chexfound_ep5[[c for c in sample_df.columns if c in submission_chexfound_ep5.columns]]


    # 
    '''
    SECTION 14: Run CheXFound Inference - Fold 6
    '''

    test_dataset = ChestXrayDataset(
        df=test_df,
        img_dir=test_images_dir,
        dataset_type="test",
        transform=test_tfms
    )

    all_ids,final_probs = predict_chexfound(
        test_dataset,
        base_model,
        model_path=f'{chexfound_folder_path}/CheXFound/ema_model_f6.pth',
        batch_size=64,
        device=DEVICE
    )

    all_preds = np.vstack(final_probs)

    # Build submission dataframe
    sample_df = test_df

    submission_chexfound_ep6 = pd.DataFrame(all_preds, columns=labels_cols)
    submission_chexfound_ep6.insert(0, "Image_name", all_ids)
    submission_chexfound_ep6 = submission_chexfound_ep6[[c for c in sample_df.columns if c in submission_chexfound_ep6.columns]]

    # Free GPU memory before loading next model
    torch.cuda.empty_cache()


    # 
    '''
    =============================================================================
    MODEL 2: EVA-X SETUP
    =============================================================================
    '''

    # 
    '''
    SECTION 15: Test-Time Augmentation (TTA) Transforms for EVA-X
    Create 5 different versions of each image for more robust predictions
    '''

    evax_img_size = 448  # EVA-X uses 448x448 images

    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def scale_brightness(factor):
        """Helper function to adjust image brightness"""
        def apply_fn(x, **kwargs):
            return (x * factor).clip(0, 255).astype(x.dtype)
        return apply_fn

    # TTA: 5 different transforms (original, flip, zoom, darker, brighter)
    tta_transforms = [

        # 1. Original image
        A.Compose([
            A.Resize(evax_img_size, evax_img_size),
            normalize,
            ToTensorV2(),
        ]),
    ]


    # 
    '''
    SECTION 16: EVA-X Prediction Function with TTA
    Runs inference with all TTA transforms and averages results
    '''

    evax_pretrained_weights = f'{evax448_folder_path}/eva_x_base_patch16_merged520k_mim.pt'

    scaler = GradScaler(device="cuda")

    def predict_tta(dataset, tta_transforms, model_path='', batch_size=128, device="cuda"):
        fold_probs = []
        all_ids = None
        logits_list = dict()  # Store raw logits
        
        # Load EVA-X model
        model = EVAX_Model(path_to_eva_x_base_patch16_pretrained_weights=evax_pretrained_weights)
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE.type,
            weights_only=False  
        )
        model.load_state_dict(checkpoint["ema_state_dict"])
        
        # Use multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"⚡ Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)  # wrap for multi-GPU
        
        model = model.to(DEVICE)
        model.eval()
        
        ids = []
        
        # Loop through each TTA transform
        for tfm in tta_transforms:
            dataset.transform = tfm  # Set transform dynamically for this TTA version 
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0 if not torch.cuda.is_available() else chosen_num_workers, pin_memory=True)
            
            fold_out = []
            with torch.no_grad():
                for images, img_id in tqdm(loader, desc="EVA-X TTA Inference"):
                    images = images.to(DEVICE)
                    with autocast(device_type=DEVICE.type):
                        logits = model(images)  # Raw predictions
                    for i, sid in enumerate(img_id):
                        logits_list[sid] = logits[i].cpu().float().numpy()
                    probs = torch.sigmoid(logits)  # Convert to probabilities [0-1]
                    fold_out.append(probs.cpu())
                    if all_ids is None:
                        ids.extend(img_id)

            # Create dataframe from dictionary of logits list and save as CSV for analysis
            logits_df = pd.DataFrame.from_dict(logits_list, orient='index')
            # If we have the label names in scope, set them as columns
            if 'labels_cols' in locals() and len(logits_df.columns) == len(labels_cols):
                logits_df.columns = labels_cols
            # Make the index an explicit column called Image_name and save without the index
            logits_df.index.name = "Image_name"
            logits_df = logits_df.reset_index()
            logits_csv_path = f'{output_folder_path}/{model_path.split("/")[-1].split(".")[0]}_fullset_evax_logits_list.csv'
            logits_df.to_csv(logits_csv_path, index=False)
            artifact = wandb.Artifact(name=f"evax_{model_path.split('/')[-1].split('.')[0]}_fullset_evax_logits", type="predictions")
            artifact.add_file(logits_csv_path)
            run.log_artifact(artifact)


            fold_out = torch.cat(fold_out, dim=0)
            fold_probs.append(fold_out)

            if all_ids is None:
                all_ids = ids

        # Average predictions from all TTA transforms
        final_probs = torch.stack(fold_probs, dim=0).mean(0).cpu()
        return all_ids, final_probs


    # 
    '''
    SECTION 17: Run EVA-X Inference with TTA - Fold 5
    Transform=None because predict_tta sets it dynamically inside
    '''

    test_dataset = ChestXrayDataset(
        df=test_df,
        img_dir=test_images_dir,
        dataset_type="test",
        transform=None  # Transform set dynamically inside predict_tta
    )

    all_ids,final_probs = predict_tta(
        test_dataset,
        tta_transforms,
        model_path=f'{evax448_folder_path}/ema_model_f5.pth',
        batch_size=128,
        device=DEVICE
    )

    all_preds = np.vstack(final_probs)

    # Build submission dataframe
    sample_df = test_df

    submission_evax448_ep5 = pd.DataFrame(all_preds, columns=labels_cols)
    submission_evax448_ep5.insert(0, "Image_name", all_ids)
    submission_evax448_ep5 = submission_evax448_ep5[[c for c in sample_df.columns if c in submission_evax448_ep5.columns]]


    # 
    '''
    SECTION 18: Run EVA-X Inference with TTA - Fold 6
    '''

    test_dataset = ChestXrayDataset(
        df=test_df,
        img_dir=test_images_dir,
        dataset_type="test",
        transform=None  # Transform set dynamically inside predict_tta
    )

    all_ids,final_probs = predict_tta(
        test_dataset,
        tta_transforms,
        model_path=f'{evax448_folder_path}/ema_model_f6.pth',
        batch_size=128,
        device=DEVICE
    )

    all_preds = np.vstack(final_probs)

    # Build submission dataframe
    sample_df = test_df

    submission_evax448_ep6 = pd.DataFrame(all_preds, columns=labels_cols)
    submission_evax448_ep6.insert(0, "Image_name", all_ids)
    submission_evax448_ep6 = submission_evax448_ep6[[c for c in sample_df.columns if c in submission_evax448_ep6.columns]]

    # Free GPU memory
    torch.cuda.empty_cache()


    # 
    '''
    =============================================================================
    FINAL ENSEMBLE - COMBINE ALL MODEL PREDICTIONS
    =============================================================================
    '''

    '''
    SECTION 19: Average All Model Predictions
    Combine predictions from:
    - CheXFound Fold 5
    - CheXFound Fold 6
    - EVA-X Fold 5 (with TTA)
    - EVA-X Fold 6 (with TTA)
    '''
    avg_sub = submission_chexfound_ep5.copy()

    # Average the 4 sets of predictions (ensemble)
    avg_sub.iloc[:, 1:] = (
        submission_chexfound_ep5.iloc[:, 1:] +
        submission_chexfound_ep6.iloc[:, 1:] +
        submission_evax448_ep5.iloc[:, 1:] +
        submission_evax448_ep6.iloc[:, 1:]
    ) / 4

    # Save final submissions
    submission_path = f'{output_folder_path}/{use_dataset}_set_submission.csv'
    avg_sub.to_csv(submission_path, index=False)
    print("✅ Final ensemble submission saved as submission.csv")

    # Log final ensemble submission as a wandb artifact
    submission_artifact = wandb.Artifact(name="ensemble_submission", type="submission")
    submission_artifact.add_file(submission_path)
    run.log_artifact(submission_artifact)
    run.finish()

    # 


if __name__ == "__main__":
    main()
