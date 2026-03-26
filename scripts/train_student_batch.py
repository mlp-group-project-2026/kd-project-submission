import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models
import timm
import sys
from pathlib import Path
import yaml
import wandb
from typing import List, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path to enable imports from project root
sys.path.append(str(Path(__file__).parent.parent))

from train_utils import train_one_epoch, validate
from loss.student_loss import DistillationLoss
from dataset import ChestXrayDataset

VALID_MACHINES = ["np", "avk", "tbsk", "eidf", "eidf_np"]



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
    CONFIG_PATH = Path(__file__).parent.parent / args.yaml
    machine_cfg, train_cfg, teachers_cfg = load_config(args.machine, CONFIG_PATH)

    print(f"\nUsing machine config: '{args.machine}'")
    print(f"  data_path: {machine_cfg['data_path']}")

    # ── Training hyperparameters from config ──
    model_name      = train_cfg.get("model_name", "mobilenet_v3_small")
    img_size        = train_cfg.get("img_size", 512)
    batch_size      = train_cfg.get("batch_size", 64)
    num_epochs      = train_cfg.get("num_epochs", 5)
    num_workers     = machine_cfg.get("num_workers", 0)
    lr              = train_cfg.get("lr", 0.001)
    optimizer_name  = train_cfg.get("optimizer", "adam")
    sched_factor    = train_cfg.get("scheduler_factor", 0.5)
    sched_patience  = train_cfg.get("scheduler_patience", 2)
    use_teacher     = train_cfg.get("use_teacher_logits", True)
    teacher_df      = args.teacher if use_teacher else None
    alpha           = train_cfg.get("alpha", 0.0)
    beta            = train_cfg.get("beta", 0.0)
    temperature     = train_cfg.get("temperature", 3.0)
    alpha_values = alpha if isinstance(alpha, list) else [alpha]
    temperature_values = temperature if isinstance(temperature, list) else [temperature]
    use_focal       = train_cfg.get("use_focal", False)
    focal_gamma     = train_cfg.get("focal_gamma", 2.0)
    train_prefetch_factor = train_cfg.get("train_prefetch_factor", 4)
    val_prefetch_factor = train_cfg.get("val_prefetch_factor", 2)
    validate_every = max(1, int(train_cfg.get("validate_every", 1)))
    use_fast_augs = train_cfg.get("use_fast_augs", False)
    use_channels_last = train_cfg.get("use_channels_last", True)
    use_torch_compile = train_cfg.get("use_torch_compile", False)
    torch_compile_mode = train_cfg.get("torch_compile_mode", "default")

    if isinstance(alpha, List):
        mode = "multi" if len(alpha) > 1 else "single"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ── Resolve teacher logits ──
    teacher_logits_df = None
    if use_teacher and teacher_df is not None:
        if teacher_df not in teachers_cfg:
            available = ", ".join(teachers_cfg.keys()) or "(none)"
            raise ValueError(
                f"Unknown teacher '{teacher_df}'. Available teachers: {available}"
            )
        teacher_arg = teachers_cfg[teacher_df]
        teacher_entry = teacher_arg if isinstance(teacher_arg, list) else [teacher_arg]
        if isinstance(teacher_entry, list):
            # Ensemble: load each CSV and store as list of DataFrames
            teacher_logits_df = [pd.read_csv(f'{machine_cfg["teacher_logits_path"]}/{csv}') for csv in teacher_entry]
            print(f"  teacher logits: {teacher_df} -> ENSEMBLE ({len(teacher_entry)} CSVs)")
        else:
            teacher_logits_csv = f'{machine_cfg["teacher_logits_path"]}/{teacher_entry}'
            print(f"  teacher logits: {teacher_df} -> {teacher_logits_csv}")
            teacher_logits_df = pd.read_csv(teacher_logits_csv)
    else:
        print("  teacher logits: None (training without distillation)")

    print()

    # Load data from machine-specific config
    data_path = machine_cfg["data_path"]
    train_csv_path = f'{data_path}/{machine_cfg["train_csv"]}'
    val_csv_path = f'{data_path}/{machine_cfg["val_csv"]}'
    img_dir = f'{data_path}/{machine_cfg["img_dir"]}'
    val_img_dir = f'{data_path}/{machine_cfg["val_img_dir"]}'
    save_dir = machine_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if use_fast_augs:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, p=0.4),
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(max_holes=1, max_height=20, max_width=20, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=0, p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create dataset and dataloader
    print("Setting up datasets and dataloaders...")
    print("Creating training dataset...")
    train_dataset = ChestXrayDataset(df=train_df, 
                                     img_dir=img_dir, 
                                     teacher_logits=teacher_logits_df,
                                     dataset_type="train",
                                     transform=transform)
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = train_prefetch_factor
    print("Creating training dataloader...")
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    print("Creating validation dataset...")

    val_dataset = ChestXrayDataset(df=val_df,
                                   img_dir=val_img_dir,
                                   dataset_type="val",
                                   transform=val_transform,
                                   teacher_logits=None)
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = val_prefetch_factor
    print("Creating validation dataloader...")
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    for alpha_run in alpha_values:
        for temperature_run in temperature_values:
            run = wandb.init(
                project="mlp-project",
                name=(
                    f"{args.machine}_{model_name}_distill-{'teacher-true' if use_teacher else 'teacher-false'}_"
                    f"alpha{alpha_run}_beta{beta}_T{temperature_run}_{optimizer_name}"
                ),
                config={
                    "machine": args.machine,
                    "model_name": model_name,
                    "img_size": img_size,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "scheduler_factor": sched_factor,
                    "scheduler_patience": sched_patience,
                    "use_teacher_logits": use_teacher,
                    "teacher_df": teacher_df,
                    "alpha": alpha_run,
                    "beta": beta,
                    "temperature": temperature_run,
                    "use_focal_loss": use_focal,
                    "focal_gamma": focal_gamma,
                    "train_prefetch_factor": train_prefetch_factor,
                    "val_prefetch_factor": val_prefetch_factor,
                    "validate_every": validate_every,
                    "use_fast_augs": use_fast_augs,
                    "use_channels_last": use_channels_last,
                    "use_torch_compile": use_torch_compile,
                    "torch_compile_mode": torch_compile_mode,
                }
            )

            #setup model
            if model_name == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(pretrained=True)
                model.classifier[3] = nn.Linear(in_features=1024, out_features=14)
            elif model_name == "mobilevit_v2_050":
                model = timm.create_model("mobilevitv2_050", pretrained=True, num_classes=14)
            else:
                raise ValueError(f"Unsupported model name: {model_name}. Choose 'mobilenet_v3_small' or 'mobilevit_v2_050'.")

            # per-run save paths
            save_path = f'{save_dir}/{model_name}_alpha{alpha_run}_T{temperature_run}_best.pth'
            loss_csv_path = f'{save_dir}/{model_name}_alpha{alpha_run}_T{temperature_run}_loss_curves.csv'

            # Setup device, loss function, and optimizer
            print("Setting up model, loss function, and optimizer...")
            device = torch.device('mps' if args.machine == 'avk' else 'cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            model = model.to(device)
            if use_channels_last and device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)
            if use_torch_compile and hasattr(torch, "compile"):
                print(f"Compiling model with torch.compile(mode='{torch_compile_mode}') ...")
                model = torch.compile(model, mode=torch_compile_mode)

            criterion = DistillationLoss(
                alpha=alpha_run, beta=beta, temperature=temperature_run,
                use_focal=use_focal, focal_gamma=focal_gamma,
            )

            # Optimizer
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            elif optimizer_name == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=sched_factor, patience=sched_patience,
            )

            # Print training config summary
            print(f"Model:       {model_name}")
            print(f"Img size:    {img_size}")
            print(f"Batch size:  {batch_size}")
            print(f"Epochs:      {num_epochs}")
            print(f"LR:          {lr}  ({optimizer_name})")
            print(f"Distill:     alpha={alpha_run}, beta={beta}, T={temperature_run}")
            print(f"Focal loss:  {'ON (gamma=' + str(focal_gamma) + ')' if use_focal else 'OFF'}")
            print(f"Teacher:     {teacher_df or 'None'}")
            print(f"Val every:   {validate_every} epoch(s)")
            print(f"Fast augs:   {'ON' if use_fast_augs else 'OFF'}")
            print(f"Compile:     {'ON' if use_torch_compile else 'OFF'}")
            print()

            # Training loop
            best_val_loss = float('inf')

            print("Starting training...")
            print("=" * 60)

            train_losses = []
            train_hard_losses = []
            train_soft_losses = []
            val_losses = []
            val_aurocs = []

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 60)
                
                # Train for one epoch
                train_loss, train_hard_loss, train_soft_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                train_losses.append(train_loss)
                train_hard_losses.append(train_hard_loss)
                train_soft_losses.append(train_soft_loss)

                should_validate = ((epoch + 1) % validate_every == 0) or (epoch + 1 == num_epochs)

                if should_validate:
                    val_loss, macro_auc, per_label_auc = validate(model, val_loader, criterion, device)
                    val_losses.append(val_loss)
                    val_aurocs.append(macro_auc)

                    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {macro_auc:.4f}")

                    # Step the scheduler based on validation loss
                    scheduler.step(val_loss)

                    # Log metrics to WandB
                    run.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_hard_loss": train_hard_loss,
                        "train_soft_loss": train_soft_loss,
                        "val_loss": val_loss,
                        "val_macro_auroc": macro_auc,
                        **{f"val_auroc_{label}": auc for label, auc in per_label_auc.items()},
                    })

                    # Save best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), save_path)
                        print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

                        # Also log the metrics of the best model to WandB
                        run.log({
                            "best_val_loss": best_val_loss,
                            "best_val_macro_auroc": macro_auc,
                            **{f"best_val_auroc_{label}": auc for label, auc in per_label_auc.items()},
                        })
                else:
                    val_losses.append(np.nan)
                    val_aurocs.append(np.nan)
                    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Validation skipped")
                    run.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_hard_loss": train_hard_loss,
                        "train_soft_loss": train_soft_loss,
                    })

            print("\n" + "=" * 60)
            print("Training completed!")
            if np.isfinite(best_val_loss):
                print(f"Best validation loss: {best_val_loss:.4f}")
            else:
                print("Best validation loss: N/A")
            print(f"Best model saved to: {save_path}")

            # Save loss curves to CSV
            loss_df = pd.DataFrame({
                "epoch": list(range(1, num_epochs + 1)),
                "train_loss": train_losses,
                "train_hard_loss": train_hard_losses,
                "train_soft_loss": train_soft_losses,
                "val_loss": val_losses,
                "val_auroc": val_aurocs,
            })
            loss_df.to_csv(loss_csv_path, index=False)
            print(f"Loss curves saved to: {loss_csv_path}")

            run.finish()

if __name__ == "__main__":
    main()
