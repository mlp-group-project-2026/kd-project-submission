'''
Code taken from: https://github.com/Masry5/Grand-X-Ray-Slam-competition and modified
'''

import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

cv2.setNumThreads(0)

class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, dataset_type, transform, teacher_logits=None):
        """
        ChestXrayDataset for medical image classification.
        
        Args:
            df: DataFrame with image names and labels
            img_dir: Directory containing images
            dataset_type: Type of dataset (e.g., "train", "val", "test")
            teacher_logits: Optional teacher logits dataframe or list of dataframes for ensemble
            transform: Transform to apply to current dataset type
        """
        self.df = df
        self.img_dir = img_dir
        self.dataset_type = dataset_type
        self.teacher_logits = teacher_logits #teacher logits dataframe, optional
        self.transform = transform
        self.label_cols = [
            "Atelectasis","Cardiomegaly","Consolidation","Edema",
            "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
            "Lung Opacity","No Finding","Pleural Effusion","Pleural Other",
            "Pneumonia","Pneumothorax","Support Devices",
        ]
        image_names_raw = df["Image_name"].tolist()
        labels_raw = df[self.label_cols].to_numpy(dtype="float32", copy=True)
        # Match teacher logits to labels by image name
        if teacher_logits is not None:
            if isinstance(teacher_logits, list):
                # Ensemble: build dict for each teacher
                teacher_logits_dicts = [df_logits.set_index("Image_name")[self.label_cols].to_dict(orient="index") for df_logits in teacher_logits]
            else:
                teacher_logits_dicts = teacher_logits.set_index("Image_name")[self.label_cols].to_dict(orient="index")
        else:
            teacher_logits_dicts = None

        # Filter out images that are missing or have 0 bytes
        valid_indices = []
        skipped = []
        for i, name in enumerate(image_names_raw):
            path = os.path.join(img_dir, name)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                skipped.append(name)
            else:
                valid_indices.append(i)
        if skipped:
            print(
                f"[ChestXrayDataset] Skipping {len(skipped)} invalid image(s) "
                f"(missing or 0-byte) out of {len(image_names_raw)} total."
            )

        self.image_names = [image_names_raw[i] for i in valid_indices]
        self.labels = labels_raw[valid_indices]
        # For teacher logits, match by image name
        if teacher_logits_dicts is not None:
            if isinstance(teacher_logits_dicts, list):
                # Ensemble: for each teacher, collect logits for valid images
                self.teacher_logits_values = [
                    np.array([list(teacher_dict[name].values()) for name in self.image_names], dtype="float32")
                    for teacher_dict in teacher_logits_dicts
                ]
            else:
                self.teacher_logits_values = np.array([
                    list(teacher_logits_dicts[name].values()) for name in self.image_names
                ], dtype="float32")
        else:
            self.teacher_logits_values = None


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset by index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: Depending on availability:
                - (img, labels): Tuple of image tensor and label tensor if teacher logits unavailable
                - (img, labels, teacher_logits): Tuple including teacher logits if available
            Where:
                - img (torch.Tensor): Image tensor of shape (C, H, W) with values in [0, 1]
                - labels (torch.Tensor): Label tensor of shape (num_labels,) with float32 dtype
                - teacher_logits (torch.Tensor, optional): Teacher model logits of shape (num_labels,)
        Raises:
            FileNotFoundError: If the image file at the specified path does not exist.
        Notes:
            - Converts grayscale images to 3-channel RGB
            - Converts RGBA images to RGB by dropping alpha channel
            - Converts OpenCV's BGR format to RGB
            - Applies augmentation transforms if provided (train_tfms for training, val_tfms for validation)
            - The permutation (2, 0, 1) converts images from HWC (Height, Width, Channel) format 
              to CHW (Channel, Height, Width) format, which is the standard for PyTorch models
        """
        img_path = os.path.join(self.img_dir, self.image_names[idx])

        # Fast read with OpenCV. It may return None if file missing -> handle.
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # If grayscale (H, W), convert to 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # if RGBA, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # OpenCV loads BGR -> convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # For test/inference: return image name instead of labels
        if self.dataset_type == "test":
            return img, self.image_names[idx]

        labels = torch.from_numpy(self.labels[idx])

        # Load teacher logits if available
        if self.teacher_logits_values is not None:
            if isinstance(self.teacher_logits_values, list):
                # Ensemble: get logits from each teacher for this sample
                teacher_logits = [torch.from_numpy(arr[idx]) for arr in self.teacher_logits_values]
                return img, labels, teacher_logits
            else:
                teacher_logits = torch.from_numpy(self.teacher_logits_values[idx])
                return img, labels, teacher_logits

        return img, labels


# choose img_size once
img_size = 512
size_tuple = (img_size, img_size)   # IMPORTANT: albumentations expects a tuple

