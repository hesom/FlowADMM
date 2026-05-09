# This file includes code derived from the upstream PnP-Flow repository:
# https://github.com/annegnx/PnP-Flow
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings

import pandas as pd
import torch
import torchvision.transforms as v2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def custom_collate(batch):
    batch = [sample for sample in batch if sample[0] is not None]
    images = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    if not images:
        return None, None
    return torch.stack(images, dim=0), labels


def _resolve_existing_dir(*candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


class DataLoaders:
    def __init__(self, dataset_name, batch_size_train, batch_size_test):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_data(self):
        if self.dataset_name == "celeba":
            transform = v2.Compose(
                [
                    v2.CenterCrop(178),
                    v2.Resize((128, 128)),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            img_dir = "./data/celeba/img_align_celeba/"
            partition_csv = "./data/celeba/list_eval_partition.csv"
            train_dataset = CelebADataset(img_dir, partition_csv, partition=0, transform=transform)
            val_dataset = CelebADataset(img_dir, partition_csv, partition=1, transform=transform)
            test_dataset = CelebADataset(img_dir, partition_csv, partition=2, transform=transform)
        elif self.dataset_name == "afhq_cat":
            transform = v2.Compose(
                [
                    v2.Resize((256, 256)),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            train_dataset = AFHQDataset(
                _resolve_existing_dir("./data/afhq_cat/train/cat/", "./.data/afhq_cat/train/cat/"),
                transform=transform,
            )
            val_dataset = AFHQDataset(
                _resolve_existing_dir("./data/afhq_cat/val/cat/", "./.data/afhq_cat/val/cat/"),
                transform=transform,
            )
            test_dataset = AFHQDataset(
                _resolve_existing_dir("./data/afhq_cat/test/cat/", "./.data/afhq_cat/test/cat/"),
                transform=transform,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            collate_fn=custom_collate,
            drop_last=(self.dataset_name == "afhq_cat"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            collate_fn=custom_collate,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            collate_fn=custom_collate,
        )
        return {"train": train_loader, "val": val_loader, "test": test_loader}


class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_csv, partition, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        partition_df = pd.read_csv(
            partition_csv, header=0, names=["image", "partition"], skiprows=1
        )
        self.img_names = partition_df[partition_df["partition"] == partition]["image"].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


class AFHQDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.files = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            warnings.warn(f"File not found: {img_path}. Skipping.")
            return None, None

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0
