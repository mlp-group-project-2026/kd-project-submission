import torch
import torch.nn as nn
import numpy as np
import copy
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from timm.layers import resample_abs_pos_embed, resample_patch_embed



class GradScalerManager:
    """Wrapper for automatic mixed precision gradient scaling."""
    def __init__(self, device="cuda"):
        self.scaler = GradScaler(device=device)
    
    def scale(self, loss):
        return self.scaler.scale(loss)
    
    def step(self, optimizer):
        self.scaler.step(optimizer)
    
    def update(self):
        self.scaler.update()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device=None,
    ema=None,
    grad_clip=1.0,
    accum_steps=1,
    scheduler=None,
    scaler=None,
):
    """
    Consolidated train loop supporting:
      - optional teacher logits in batches (batch length 3)
      - AMP via `autocast` + `GradScaler`
      - gradient accumulation (`accum_steps`)
      - optional LR `scheduler` and EMA updates

    Args:
        model, loader, optimizer, criterion: usual training objects
        device: `torch.device` or None (will infer if None)
        ema: ModelEMA instance (optional)
        grad_clip: max norm for gradient clipping (optional)
        accum_steps: accumulate gradients for N mini-batches
        scheduler: LR scheduler to step when optimizer steps
        scaler: optional external GradScaler

    Returns:
        Average training loss over dataset
    """
    model.train()
    running_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a scaler if not provided
    if scaler is None:
        try:
            scaler = GradScaler(device="cuda")
        except TypeError:
            scaler = GradScaler()

    optimizer.zero_grad()
    total_batches = len(loader)

    progress_bar = tqdm(enumerate(loader), total=total_batches, desc="[Train]")
    for batch_idx, batch in progress_bar:
        # support (imgs, labels) or (imgs, labels, teacher_logits)
        if len(batch) == 3:
            imgs, labels, teacher_logits = batch
            if isinstance(teacher_logits, list):
                teacher_logits = [t.to(device).float() for t in teacher_logits]
            else:
                teacher_logits = teacher_logits.to(device).float()
        else:
            imgs, labels = batch
            teacher_logits = None

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            outputs = model(imgs)
            if teacher_logits is not None:
                loss, hard_loss, soft_loss = criterion(outputs, labels, teacher_logits)
            else:
                loss = criterion(outputs, labels)

        # scale loss for accumulation
        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if teacher_logits is not None:
            hard_loss = hard_loss / accum_steps

            soft_loss = soft_loss / accum_steps

        # perform optimizer step every `accum_steps`
        if (batch_idx + 1) % accum_steps == 0:
            if grad_clip is not None and grad_clip > 0:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_(get_model(model).parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

        # accumulate running loss (restore sample-wise loss)
        running_loss += loss.item() * imgs.size(0) * accum_steps
        if teacher_logits is not None:
            running_hard_loss += hard_loss.item() * imgs.size(0) * accum_steps
            running_soft_loss += soft_loss.item() * imgs.size(0) * accum_steps

    # handle remainder if total batches not divisible by accum_steps
    remainder = len(loader) % accum_steps
    if remainder != 0:
        if grad_clip is not None and grad_clip > 0:
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(get_model(model).parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

    avg_loss = running_loss / len(loader.dataset)
    if teacher_logits is not None:
        avg_hard_loss = running_hard_loss / len(loader.dataset)
        avg_soft_loss = running_soft_loss / len(loader.dataset)
        return avg_loss, avg_hard_loss, avg_soft_loss
    else:
        return avg_loss, None, None


def validate(model, loader, criterion, device):
    """
    Validate model and compute metrics.
    
    Args:
        model: Model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, macro AUROC, per-label AUROC dict)
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_outputs = [], []
    
    progress_bar = tqdm(loader, desc="[Val]")
    with torch.no_grad():
        for batch in progress_bar:
            # Handle both cases: with and without teacher logits
            if len(batch) == 3:
                imgs, labels, teacher_logits = batch
                imgs = imgs.to(device)
                labels = labels.to(device).float()
                teacher_logits = teacher_logits.to(device).float()
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device).float()
                teacher_logits = None
            
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(imgs)
                if teacher_logits is not None:
                    loss = criterion(outputs, labels, teacher_logits)
                else:
                    loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.sigmoid().cpu().numpy())
    
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    # Compute per-label AUROC
    per_label_auc = {}
    label_cols = getattr(loader.dataset, 'label_cols', None)
    
    if label_cols is not None:
        for i, label in enumerate(label_cols):
            try:
                per_label_auc[label] = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            except ValueError:
                per_label_auc[label] = np.nan
    
    # Compute macro AUROC
    try:
        macro_auc = roc_auc_score(all_labels, all_outputs, average="macro")
    except ValueError:
        macro_auc = np.nan
    
    return running_loss / len(loader.dataset), macro_auc, per_label_auc


def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """
    Create cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of training for warmup (default 0.1)
        
    Returns:
        Learning rate scheduler
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return scheduler


def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


def get_model(wrapped_model):
    """
    Extract underlying model from DataParallel wrapper if present.
    
    Args:
        wrapped_model: Model possibly wrapped with nn.DataParallel
        
    Returns:
        Underlying model
    """
    return getattr(wrapped_model, "module", wrapped_model)


class ModelEMA:
    """Exponential Moving Average of model weights for semi-supervised learning."""
    
    def __init__(self, model, decay=0.9997, device=None):
        """
        Initialize EMA model.
        
        Args:
            model: Model to create EMA from
            decay: Decay rate for exponential moving average (default 0.9997)
            device: Device to place EMA model on (optional)
        """
        # copy *unwrapped* module (so EMA keys match saved/loaded model keys)
        self.decay = decay
        self.device = device
        self.ema = copy.deepcopy(get_model(model)).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Update EMA model weights.
        
        Args:
            model: Current model to update EMA from
        """
        src_state = get_model(model).state_dict()
        with torch.no_grad():
            for k, v in self.ema.state_dict().items():
                model_v = src_state[k].detach().to(v.device)
                v.copy_(v * self.decay + (1.0 - self.decay) * model_v)

    def state_dict(self):
        """Return EMA model state dictionary."""
        return self.ema.state_dict()
