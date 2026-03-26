import os
import sys
import torch
import torch.nn as nn
import json

# Setup paths for imports
script_dir = os.path.dirname(__file__)
code_root = os.path.join(script_dir, '..')
chexfound_root = os.path.join(code_root, 'model', 'chexfound', 'CheXFound')

sys.path.insert(0, chexfound_root)  # For chexfound package

from chexfound.eval.setup import setup_and_build_model
from chexfound.eval.utils import extract_hyperparameters_from_model
from chexfound.eval.classification.utils import setup_glori
from fvcore.common.checkpoint import Checkpointer


class CheXFoundWithGLoRIHead(nn.Module):
    """Wrapper combining CheXFound backbone with GLoRI classification head."""
    def __init__(self, backbone, glori_module, args):
        super().__init__()
        self.backbone = backbone
        self.glori = glori_module
        self.args = args
    
    def forward(self, x):
        # get intermediate features and pass to glori
        feats = self.backbone.get_intermediate_layers(
            x, 
            n=self.args.n_last_blocks, 
            return_class_token=self.args.return_class_token
        )
        logits_dict = self.glori([feats], return_attention=False)
        key = list(logits_dict.keys())[0]
        return logits_dict[key]


def setup_foundation_model(args, pretrained_weights, device=None):
    """
    Initialize and load the foundation model (CheXFound backbone).
    
    Args:
        args: Arguments containing model configuration
        pretrained_weights: Path to pretrained model checkpoint
        device: Device to use (default: cuda if available else cpu)
        
    Returns:
        base_model: Loaded foundation model
        autocast_dtype: Data type for automatic casting
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model, autocast_dtype = setup_and_build_model(args)
    base_model = base_model.to(device)
    
    # Load checkpoint for foundation model with proper device mapping
    state_dict = torch.load(pretrained_weights, map_location=device)['teacher']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone'):
            ls = k.split('.')
            if 'blocks' in k:
                new_k = '.'.join([ls[1], *ls[3:]])
            else:
                new_k = '.'.join(ls[1:])
        else:
            new_k = k
        new_state_dict.update({new_k: v})
    
    base_model.load_state_dict(new_state_dict, strict=False)
    return base_model, autocast_dtype


def setup_glori_head(args, base_model, classifier_json, classifier_fpath, device=None):
    """
    Initialize and load the GLoRI classification head.
    
    Args:
        args: Arguments containing model configuration
        base_model: Foundation model (backbone)
        classifier_json: Path to classifier configuration JSON
        classifier_fpath: Path to pretrained GLoRI classifier checkpoint
        device: Device to use (default: cuda if available else cpu)
        
    Returns:
        glori: Initialized GLoRI module with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract hyperparameters from best classifier
    log_json = classifier_json
    with open(log_json, 'r') as f:
        content = f.read().split('\n')[-3]
        data = json.loads(content)
    best_classifier_str = data['best_classifier']['name']
    hyperparameters = extract_hyperparameters_from_model(best_classifier_str)
    learning_rate = hyperparameters["lr"]
    avgpool = hyperparameters["avgpool"]
    block = hyperparameters["blocks"]
    
    # Get sample output from backbone
    sample_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    with torch.no_grad():
        sample_output = base_model.get_intermediate_layers(
            sample_input, 
            n=args.n_last_blocks, 
            return_class_token=True
        )
    
    # Setup GLoRI
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
    
    # Move glori to device before loading checkpoint
    glori = glori.to(device)
    
    # Load checkpoint for glori classifier
    checkpointer = Checkpointer(glori)
    checkpointer.load(classifier_fpath)
    
    return glori


def resize_decoder_for_num_classes(glori, target_classes, orig_classes=40, device=None):
    """
    Resize decoder parameters and biases to match target number of classes.
    
    Args:
        glori: GLoRI module
        target_classes: Target number of output classes
        orig_classes: Original number of classes (default 40)
        device: Device to use for new tensors (default: cuda if available else cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get classifier from GLoRI
    glori_module = getattr(glori, "module", glori)
    classifier_key = next(iter(glori_module.classifiers_dict.keys()))
    classifier = glori_module.classifiers_dict[classifier_key]
    decoder = getattr(classifier, "decoder", None)
    
    if decoder is None:
        raise RuntimeError("Decoder not found in classifier; cannot resize num_classes.")
    
    print("Decoder found:", type(decoder))
    print("Decoder.num_classes (before):", getattr(decoder, "num_classes", None))
    
    # 1) Set decoder.num_classes
    decoder.num_classes = target_classes
    print("Set decoder.num_classes ->", decoder.num_classes)
    
    # 2) Replace duplicate_pooling_bias if exists
    if hasattr(decoder, "duplicate_pooling_bias"):
        old = decoder.duplicate_pooling_bias
        print("Old duplicate_pooling_bias shape:", 
              tuple(old.shape) if isinstance(old, torch.Tensor) else type(old))
        new_bias = nn.Parameter(torch.zeros(target_classes, dtype=torch.float32, device=device))
        nn.init.constant_(new_bias, 0.0)
        decoder.duplicate_pooling_bias = new_bias
        print("Replaced duplicate_pooling_bias with shape:", decoder.duplicate_pooling_bias.shape)
    else:
        print("No attribute duplicate_pooling_bias found on decoder — fine.")
    
    # 3) Resize 1D parameters matching original class count
    replaced = []
    for name, param in list(decoder.named_parameters()):
        if param.dim() == 1 and param.shape[0] == orig_classes:
            print(f"Resizing param: {name} shape {param.shape} -> {target_classes}")
            parent = decoder
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            new_p = nn.Parameter(torch.zeros(target_classes, dtype=param.dtype, device=device))
            setattr(parent, attr, new_p)
            replaced.append(name)
    
    # 4) Resize buffers (non-parameter tensors)
    for name, buf in list(decoder.named_buffers()):
        if buf is None:
            continue
        if buf.ndim == 1 and buf.shape[0] == orig_classes:
            print(f"Resizing buffer: {name} shape {buf.shape} -> {target_classes}")
            parent = decoder
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            decoder.register_buffer(attr, torch.zeros(target_classes, dtype=buf.dtype, device=device))
            replaced.append("buffer:"+name)
    
    print("Replaced parameter/buffer names:", replaced)
    
    # 5) Move whole glori to device
    glori.to(device)
    print("Moved glori to device.")


def prepare_model_for_training(base_model, glori, args, freeze_backbone=True, use_data_parallel=True, device=None):
    """
    Wrap and prepare model for training (backbone + GLoRI head).
    
    Args:
        base_model: Foundation model (backbone)
        glori: GLoRI classification head
        args: Arguments containing model configuration
        freeze_backbone: Whether to freeze backbone parameters
        use_data_parallel: Whether to wrap with nn.DataParallel for multi-GPU
        device: Device to use (default: cuda if available else cpu)
        
    Returns:
        model: Wrapped model ready for training
        device: Device model is on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Preparing model on device: {device}, GPUs: {torch.cuda.device_count()}")
    
    # 1) Unwrap DDP if necessary
    if hasattr(glori, "module"):
        print("Unwrapping glori from DDP wrapper -> using glori.module")
        glori_clean = glori.module
    else:
        glori_clean = glori
    
    # Move to CPU temporarily to avoid device mismatch
    glori_clean = glori_clean.cpu()
    
    # 2) Create training model
    model = CheXFoundWithGLoRIHead(backbone=base_model, glori_module=glori_clean, args=args)
    
    # 3) Freeze backbone if requested, ensure glori params trainable
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    
    for p in model.glori.parameters():
        p.requires_grad = True
    
    # 4) Wrap with DataParallel for multi-GPU
    if use_data_parallel and torch.cuda.device_count() > 1:
        print("Wrapping model with nn.DataParallel for multi-GPU")
        model = nn.DataParallel(model)
    
    # 5) Move to device
    model = model.to(device)
    
    return model, device


def get_glori_parameters(model):
    """
    Extract trainable GLoRI parameters from model (handles DataParallel wrapping).
    
    Args:
        model: Training model (possibly wrapped with DataParallel)
        
    Returns:
        List of trainable glori parameters
    """
    core = getattr(model, "module", model)
    glori = getattr(core, "glori", None)
    if glori is None:
        raise RuntimeError("model has no .glori attribute; inspect model structure.")
    glori_core = getattr(glori, "module", glori)
    return [p for p in glori_core.parameters() if p.requires_grad]
