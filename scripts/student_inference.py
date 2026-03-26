import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import timm
import sys
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add parent directory to path to enable imports from project root
sys.path.append(str(Path(__file__).parent.parent))

from dataset import ChestXrayDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a student model.")
    
    # Required arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., mobilevit_v2_050_tuning)")
    parser.add_argument("--expt_folder", type=str, required=True, help="Experiment folder relative to project root (e.g., experiments_eidf/student_baselines/mobilevit)")
    parser.add_argument("--arch", type=str, required=True, choices=["mobilenet_v3_small", "mobilevit_v2_050"], help="Model architecture")

    # Optional path arguments
    parser.add_argument("--data_path", type=str, default='/Users/s1807328/Desktop/MLP Project/xray-slam-data/grand-xray-slam-division-b', help="Base path for data")
    parser.add_argument("--csv_file", type=str, default='val2.csv', help="Name of the validation CSV file")
    parser.add_argument("--img_subdir", type=str, default='val2', help="Name of the validation image subdirectory")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # ========================== CONFIGURATION ==========================
    # Specify the experiment path (relative to project root)
    arch_name = args.arch
    model_name = args.model_name
    expt_folder = args.expt_folder

    # Data paths - keep or override as needed
    data_path = args.data_path
    csv_path = f'{data_path}/{args.csv_file}'
    img_dir = f'{data_path}/{args.img_subdir}'

    # Auto-infer paths based on expt_folder
    project_root = Path(__file__).parent.parent
    output_dir = project_root / expt_folder / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint path
    checkpoint_path = output_dir / f"{model_name}_best.pth"

    # Output path for inference logits
    output_path = output_dir / f"{model_name}_logits.csv"
    # ===================================================================

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Saving logits to: {output_path}")
    
    if arch_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = nn.Linear(in_features=1024, out_features=14)

    elif arch_name == "mobilevit_v2_050":
        model = timm.create_model(
            "mobilevitv2_050",
            pretrained=False,
            num_classes=14
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name} (derived from {model_name})")

    # Load trained weights
    device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    df = pd.read_csv(csv_path)

    # Inference transform: no augmentation, just resize + normalize
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = ChestXrayDataset(
        df=df,
        img_dir=img_dir,
        dataset_type="val",
        transform=transform,
        teacher_logits=None
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,  # Keep order to match CSV rows
        num_workers=0,
        pin_memory=True
    )

    # Run inference
    label_cols = dataset.label_cols
    all_logits = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Inference]"):
            imgs, labels = batch
            imgs = imgs.to(device)

            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())

    all_logits = np.vstack(all_logits)

    # Save logits to CSV — use dataset.image_names (matches actual inference count,
    # since the dataset filters out missing/corrupt images in __init__)
    logits_df = pd.DataFrame(all_logits, columns=label_cols)
    logits_df.insert(0, "Image_name", dataset.image_names[:len(all_logits)])

    output_path.parent.mkdir(exist_ok=True)
    logits_df.to_csv(output_path, index=False)

    print(f"\nLogits saved to {output_path}")
    print(f"Shape: {all_logits.shape}")
    print(logits_df.head())


if __name__ == "__main__":
    main()
