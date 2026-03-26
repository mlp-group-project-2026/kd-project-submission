FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip --root-user-action=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    numpy==1.26.4 \
    pandas \
    timm \
    albumentations \
    scikit-learn \
    transformers \
    scikit-image \
    opencv-python \
    wandb

WORKDIR /data
ENTRYPOINT ["/bin/bash", "-lc"]
