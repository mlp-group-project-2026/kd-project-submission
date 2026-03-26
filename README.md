# Ensemble KD for Chest X-Ray Multi-Label Classification

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)

This project implements an Ensemble Knowledge Distillation (KD) pipeline for chest X-ray multi-label classification, inspired by the Kaggle Grand Slam X-Ray challenge. The workflow distills knowledge from stronger teacher predictions (CheXFound and EVA-X logits, including ensembles) into lightweight student models for 14-pathology prediction.

The codebase is organized around scripts in the `scripts/` folder, with a Docker environment available via `Dockerfile`.

## Project Scope

- Task: Multi-label chest X-ray classification.
- Labels: 14 findings (Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices).
- Student architectures: `mobilenet_v3_small`, `mobilevit_v2_050`.
- Distillation: teacher logits from CheXFound, EVA-X, or teacher ensemble.
- Evaluation: per-class AUROC, macro AUROC, and bootstrap confidence intervals.

## Repository Layout

```text
.
|-- config.yaml
|-- Dockerfile
|-- loss/
|   |-- loss_chexfound.py
|   `-- student_loss.py
`-- scripts/
        |-- comparison.py
        |-- dataset.py
        |-- ensemble_evaluation.py
        |-- eval_utils.py
        |-- evaluation.py
        |-- graph_creations.py
        |-- inference.py
        |-- model_initialization.py
        |-- optimise_ensemble.py
        |-- student_inference.py
        |-- test_env.py
        |-- train_chexfound.py
        |-- train_evax.py
        |-- train_student.py
        |-- train_student_batch.py
        `-- train_utils.py
```

## Prerequisites

- Python 3.10
- CUDA 12.4 target environment (GPU recommended)
- Debian/Ubuntu-like Linux environment (recommended for parity)

## Dataset

This work is inspired by the Kaggle Grand Slam X-Ray challenge.

- Challenge page: https://www.kaggle.com/competitions/grand-xray-slam-division-a 
- Configure your local data paths in `config.yaml` under `machines.<your_machine>`.

Expected config-driven dataset fields include:

- `data_path`
- `train_csv`, `val_csv`
- `img_dir`, `val_img_dir`
- inference paths such as `inference_train_df`, `inference_val_df`, `inference_test_df`

## Configuration

`config.yaml` controls:

- training hyperparameters (`training` section),
- teacher shortcut names (`teachers` section),
- machine-specific paths and runtime settings (`machines` section).

Teacher shortcuts (for `-t/--teacher`) are resolved from `teachers` in `config.yaml`.

Example values currently include shortcuts like `chexfound`, `evax`, `ensemble`, and `synthetic`.

## Local Setup

1. Create and activate a Python 3.10 environment.
2. Install dependencies.

```bash
pip install --upgrade pip
pip install \
    torch==2.4.1 torchvision==0.19.1 \
    numpy==1.26.4 pandas timm albumentations scikit-learn \
    transformers scikit-image opencv-python wandb
```

3. Run the environment smoke test:

```bash
python scripts/test_env.py
```

## Docker Setup

Build image:

```bash
docker build -t kd-xray:latest .
```

Run container with GPU and mounts:

```bash
docker run --rm -it --gpus all \
    -v /path/to/xray-data:/data/xray-data \
    -v /path/to/outputs:/data/outputs \
    kd-xray:latest
```

Then execute commands from inside the container, for example:

```bash
python scripts/train_student.py -m eidf -y config.yaml -t ensemble
```

Note: the current Docker base image uses CUDA 12.1 runtime (`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`) while project prerequisites mention CUDA 12.4. Keep host driver/runtime compatibility in mind.

## Quick Start

Single student training run with ensemble teacher logits:

```bash
python scripts/train_student.py -m eidf -y config.yaml -t ensemble
```

Batch sweep over configured alpha/temperature lists:

```bash
python scripts/train_student_batch.py -m eidf -y config.yaml -t ensemble
```

## Training Workflows

### Student Training

- Script: `scripts/train_student.py`
- Required args:
    - `-m, --machine` (choices include `np`, `avk`, `tbsk`, `eidf`, `eidf_np`)
    - `-y, --yaml` (path to config file)
- Optional:
    - `-t, --teacher` (teacher shortcut from config)

Example:

```bash
python scripts/train_student.py -m eidf -y config.yaml -t chexfound
```

### Batch Student Training

- Script: `scripts/train_student_batch.py`
- Supports multi-run sweeps via list-valued config entries such as `alpha` and `temperature`.

Example:

```bash
python scripts/train_student_batch.py -m eidf -y config.yaml -t evax
```

### Teacher Training

- `scripts/train_chexfound.py`
- `scripts/train_evax.py`

Use these when you need to retrain teacher models rather than consume precomputed teacher logits.

## Inference

### Ensemble Teacher Inference (CheXFound + EVA-X)

- Script: `scripts/inference.py`
- Required args:
    - `-m, --machine`
    - `-y, --yaml`
    - `-d, --use_dataset` (`train`, `val`, `test`)

Example:

```bash
python scripts/inference.py -m eidf -y config.yaml -d val
```

### Student Checkpoint Inference

- Script: `scripts/student_inference.py`
- Required args:
    - `--model_name`
    - `--expt_folder`
    - `--arch` (`mobilenet_v3_small` or `mobilevit_v2_050`)

Example:

```bash
python scripts/student_inference.py \
    --model_name mobilevit_v2_050_alpha0.5_T4.0 \
    --expt_folder experiments_eidf/student_runs/mobilevit \
    --arch mobilevit_v2_050
```

## Evaluation

### Student or Single-Model Evaluation

- Script: `scripts/evaluation.py`
- Required args:
    - `--model_name`
    - `--expt_folder`
- Optional:
    - `--data_path`
    - `--csv_file`

Example:

```bash
python scripts/evaluation.py \
    --model_name mobilevit_v2_050_alpha0.5_T4.0 \
    --expt_folder experiments_eidf/student_runs/mobilevit \
    --data_path /path/to/grand-xray-slam-division-b \
    --csv_file val2.csv
```

### Ensemble Evaluation Utility

- Script: `scripts/ensemble_evaluation.py`
- Configure logit file paths directly inside the script, then run:

```bash
python scripts/ensemble_evaluation.py
```

## Script Guide

- `scripts/train_student.py`: Main single-run student KD training entry point.
- `scripts/train_student_batch.py`: Batch/sweep KD training over hyperparameter lists.
- `scripts/inference.py`: Teacher-model inference and ensemble logits generation.
- `scripts/student_inference.py`: Student checkpoint inference to logits CSV.
- `scripts/evaluation.py`: AUROC evaluation with bootstrap confidence intervals.
- `scripts/ensemble_evaluation.py`: Ensemble-level evaluation.
- `scripts/optimise_ensemble.py`: Ensemble weighting optimization experiments.
- `scripts/comparison.py`: Comparative analysis utilities.
- `scripts/graph_creations.py`: Plot and figure generation.
- `scripts/dataset.py`: Dataset and loading logic.
- `scripts/train_utils.py`, `scripts/eval_utils.py`: Training/evaluation shared utilities.
- `scripts/model_initialization.py`: Model setup helpers.
- `loss/student_loss.py`, `loss/loss_chexfound.py`: KD and related losses.

## Related Notebooks

From Kaggle winners' workflows used as inspiration:

1. Preprocessing: `600-p-div-a.ipynb`
2. EVA-X base model: `evax-recs-448-div-a.ipynb`
3. CheXFound model:
     - `all-data-chexfoundrecs-div-a_3.ipynb`
     - `all-data-chexfound-recs-div-a_6.ipynb`
4. Inference and ensembling: `all-models-inference-div-a.ipynb`

## Troubleshooting

- Config machine key error: ensure `-m` matches a key under `machines` in `config.yaml`.
- Missing teacher logits: ensure `-t` matches `teachers` in `config.yaml` and files exist under `teacher_logits_path`.
- File-not-found errors: confirm all CSV/image folders in `config.yaml` are valid on your machine.
- CUDA issues in Docker: verify host NVIDIA driver compatibility and Docker GPU runtime setup.
- W&B issues: login with `wandb login` or disable logging in code if running offline.

## License

See `LICENSE`.