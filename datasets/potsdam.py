import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from transformations import get_transformations, Transformation
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class PotsdamDataModule(LightningDataModule):
    def __init__(self, cfg: Dict):
        super().__init__()

        self.cfg = cfg
        self.task = cfg["model"]["task"]
        self.data_indices = np.array([0])
        self.all_indices = None

    def setup(self, stage: str = None):
        path_to_training_dataset = os.path.join(
            self.cfg["data"]["path_to_dataset"], "training_set"
        )
        path_to_validation_dataset = os.path.join(
            self.cfg["data"]["path_to_dataset"], "validation_set"
        )
        path_to_test_dataset = os.path.join(
            self.cfg["data"]["path_to_dataset"], "test_set"
        )

        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train = PotsdamDataset(
                path_to_training_dataset,
                self.cfg,
                transformations=get_transformations(self.cfg, "train"),
            )
            self._val = PotsdamDataset(
                path_to_validation_dataset,
                self.cfg,
                transformations=get_transformations(self.cfg, "val"),
            )

        if stage == "test" or stage is None:
            self._test = PotsdamDataset(
                path_to_test_dataset,
                self.cfg,
                transformations=get_transformations(self.cfg, "test"),
            )

    def append_data_indices(self, indices: np.array):
        self.data_indices = np.unique(np.append(self.data_indices, indices))

    def get_data_indices(self) -> np.array:
        return self.data_indices

    def get_unlabeled_data_indices(self) -> np.array:
        msk = ~np.in1d(self.all_indices, self.data_indices)

        return self.all_indices[msk]

    def unlabeled_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        train_dataset = self._train
        unlabeled_data = Subset(train_dataset, self.get_unlabeled_data_indices())

        loader = DataLoader(
            unlabeled_data, batch_size=batch_size, shuffle=False, num_workers=n_workers
        )

        return loader

    def train_dataloader(self) -> DataLoader:
        shuffle = self.cfg["data"]["train_shuffle"]
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        train_dataset = self._train

        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._val, batch_size=batch_size, num_workers=n_workers, shuffle=False
        )

        return loader

    def test_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._test, batch_size=batch_size, num_workers=n_workers, shuffle=False
        )

        return loader


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".jpg", ".png"])


class PotsdamDataset(Dataset):
    def __init__(
        self, path_to_dataset: str, cfg: Dict, transformations: List[Transformation]
    ):
        super().__init__()

        self.task = cfg["model"]["task"]
        self.cfg = cfg

        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError(f"Cannot find dataset at {path_to_dataset}")

        # get path to all RGB images
        self.path_to_images = os.path.join(path_to_dataset, "image")
        if not os.path.exists(self.path_to_images):
            raise FileNotFoundError(f"Cannot find images at {self.path_to_images}")

        self.image_files = []
        for fname in os.listdir(self.path_to_images):
            if is_image(fname):
                self.image_files.append(fname)
        self.image_files.sort()

        # get path to all ground-truth annotations
        self.path_to_annos = os.path.join(path_to_dataset, "anno")
        if not os.path.exists(self.path_to_annos):
            raise FileNotFoundError(f"Cannot find labels at {self.path_to_annos}")

        self.anno_files = []
        for fname in os.listdir(self.path_to_annos):
            if is_image(fname):
                self.anno_files.append(fname)
        self.anno_files.sort()

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations

    def __getitem__(self, idx: int) -> Dict:
        path_to_current_img = os.path.join(self.path_to_images, self.image_files[idx])
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)

        path_to_current_anno = os.path.join(self.path_to_annos, self.anno_files[idx])
        anno = self.get_anno(path_to_current_anno)

        # apply a set of transformations to the raw_image, image and anno
        for transformer in self.transformations:
            img_pil, img, anno = transformer(img_pil, img, anno)

        if self.task == "classification":
            anno = self.remap_annotation(anno.numpy())

        return {"data": img, "image": img, "anno": anno, "index": idx}

    def __len__(self) -> int:
        return len(self.image_files)

    def get_anno(self, path_to_current_anno: str) -> torch.Tensor:
        if self.task == "classification":
            anno = cv2.imread(path_to_current_anno)
            anno = anno.astype(np.int64)  # torch does not support conversion of uint16
            anno = np.moveaxis(anno, -1, 0)  # now in CHW mode
            return torch.from_numpy(anno).long()
        elif self.task == "regression":
            anno = cv2.imread(
                path_to_current_anno, cv2.IMREAD_GRAYSCALE
            )  # 0 to 255 value range
            anno = anno.astype(
                np.float32
            )  # torch does not support conversion of uint16, now in HW mode
            anno = (self.cfg["model"]["value_range"]["max_value"] / 255) * anno
            return torch.from_numpy(anno).float().unsqueeze(dim=0)
        else:
            raise NotImplementedError(
                f"{self.task} task is not implemented for Potsdam dataset!"
            )

    @staticmethod
    def remap_annotation(anno: np.array) -> torch.Tensor:
        """
        After remapping the annotations have the labels:

        potsdam:
            - 0 := boundary line,
            - 1 := imprevious surfaces,
            - 2 := building,
            - 3 := low vegetation,
            - 4 := tree,
            - 5 := car,
            - 6 := clutter/background
        """

        dims = anno.shape
        assert len(dims) == 3, "wrong matrix dimension!!!"
        assert dims[0] == 3, "annotation must have 3 channels!!!"
        remapped = np.zeros((dims[1], dims[2]))

        mask1 = anno[0, :, :] == 255  # B = 255
        mask2 = anno[1, :, :] == 255  # G = 255
        mask3 = anno[2, :, :] == 255  # R = 255
        mask4 = anno[0, :, :] == 0  # B = 0
        mask5 = anno[1, :, :] == 0  # G = 0
        mask6 = anno[2, :, :] == 0  # R = 0

        mask_surface = mask1 * mask2 * mask3
        remapped[mask_surface] = 1

        mask_building = mask1 * mask5 * mask6
        remapped[mask_building] = 2

        mask_veg = mask1 * mask2 * mask6
        remapped[mask_veg] = 3

        mask_tree = mask2 * mask4 * mask6
        remapped[mask_tree] = 4

        mask_car = mask2 * mask3 * mask4
        remapped[mask_car] = 5

        mask_bg = mask3 * mask4 * mask5
        remapped[mask_bg] = 6

        return torch.from_numpy(remapped).long()
