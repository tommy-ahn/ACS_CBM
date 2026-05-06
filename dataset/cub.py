# dataset/cub.py

import os
import pickle
from typing import Optional, List

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


OLD_CUB_PREFIX = "/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets"


def resolve_img_path(img_path: str, root: str) -> str:
    root = os.path.abspath(root)

    if img_path.startswith(OLD_CUB_PREFIX):
        img_path = img_path.replace(OLD_CUB_PREFIX, root)

    if os.path.exists(img_path):
        return img_path

    candidate = os.path.join(root, img_path)
    if os.path.exists(candidate):
        return candidate

    if "CUB_200_2011" in img_path:
        tail = img_path[img_path.find("CUB_200_2011") :]
        candidate = os.path.join(root, tail)
        if os.path.exists(candidate):
            return candidate

    if "images/" in img_path.replace("\\", "/"):
        tail = img_path.replace("\\", "/").split("images/", 1)[1]
        candidate = os.path.join(root, "CUB_200_2011", "images", tail)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve image path: {img_path}")


def create_transform(cfg, task="train"):
    if task.lower() == "train":
        return T.Compose(
            [
                T.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                T.RandomResizedCrop(cfg.DATASET.IMG_SIZE),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=cfg.DATASET.PIXEL_MEAN, std=cfg.DATASET.PIXEL_STD),
            ]
        )
    return T.Compose(
        [
            T.CenterCrop(cfg.DATASET.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.DATASET.PIXEL_MEAN, std=cfg.DATASET.PIXEL_STD),
        ]
    )


def load_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def find_class_imbalance_from_gt(data):
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    n_ones = [0] * n_attr

    for item in data:
        labels = item["attribute_label"]
        for i in range(n_attr):
            n_ones[i] += labels[i]

    imbalance = []
    for i in range(n_attr):
        denom = max(n_ones[i], 1)
        imbalance.append(n / denom - 1)

    return torch.FloatTensor(imbalance)


class CUB(Dataset):
    """
    Supports:
      - concept_source="gt"
      - concept_source="pseudo"

    Returns:
      image, class_label, concept_label
    """

    def __init__(
        self,
        cfg,
        task: str = "train",
        uncertain_label: bool = False,
        selected_concepts: Optional[List[int]] = None,
        concept_source: str = "gt",
        pseudo_dir: Optional[str] = None,
        pseudo_type: str = "soft",              # "soft" | "binary_threshold" | "binary_topk"
        pseudo_threshold: float = 0.28,
        pseudo_topk: int = 20,
        pseudo_name_path: Optional[str] = None,
    ):
        super().__init__()
        self.root = cfg.DATASET.ROOT
        self.task = task
        self.uncertain_label = uncertain_label
        self.selected_concepts = selected_concepts
        self.concept_source = concept_source.lower()
        self.transform = create_transform(cfg, task)

        pkl_path = os.path.join(self.root, "CUB_200_2011", f"{task}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing split file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.pseudo_concepts = None
        self.concept_names = None

        if self.concept_source == "gt":
            sample = self.data[0]
            n_concepts = len(sample["attribute_label"])
            self.concept_names = [f"gt_att_{i}" for i in range(n_concepts)]
            self.imbalance = (
                find_class_imbalance_from_gt(self.data)
                if task.lower() == "train" and cfg.TRAIN.IMBALANCE
                else None
            )

        elif self.concept_source == "pseudo":
            if pseudo_dir is None:
                raise ValueError("concept_source='pseudo' requires pseudo_dir")

            split_dir = os.path.join(pseudo_dir, task)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Missing pseudo split dir: {split_dir}")

            if pseudo_type == "soft":
                pseudo_path = os.path.join(split_dir, "soft_pseudo_gt.pt")
            elif pseudo_type == "binary_threshold":
                pseudo_path = os.path.join(split_dir, f"binary_pseudo_gt_threshold_{pseudo_threshold:.2f}.pt")
            elif pseudo_type == "binary_topk":
                pseudo_path = os.path.join(split_dir, f"binary_pseudo_gt_topk_{pseudo_topk}.pt")
            else:
                raise ValueError(f"Unknown pseudo_type: {pseudo_type}")

            if not os.path.exists(pseudo_path):
                raise FileNotFoundError(f"Missing pseudo concept file: {pseudo_path}")

            self.pseudo_concepts = torch.load(pseudo_path, map_location="cpu").float()
            if len(self.pseudo_concepts) != len(self.data):
                raise ValueError(
                    f"Pseudo concept length mismatch: pseudo={len(self.pseudo_concepts)}, data={len(self.data)}"
                )

            if pseudo_name_path is None:
                pseudo_name_path = os.path.join(split_dir, "concepts.txt")

            if pseudo_name_path is not None and os.path.exists(pseudo_name_path):
                self.concept_names = load_text_lines(pseudo_name_path)
            else:
                self.concept_names = [f"pseudo_att_{i}" for i in range(self.pseudo_concepts.shape[1])]

            self.imbalance = None

        else:
            raise ValueError(f"Unknown concept_source: {self.concept_source}")

        if self.selected_concepts is not None:
            self.concept_names = [self.concept_names[i] for i in self.selected_concepts]

    def __len__(self):
        return len(self.data)

    def _subsample(self, x):
        if self.selected_concepts is None:
            return x
        if torch.is_tensor(x):
            return x[self.selected_concepts]
        return torch.tensor(x, dtype=torch.float32)[self.selected_concepts]

    def _load_image(self, idx):
        img_path = resolve_img_path(self.data[idx]["img_path"], self.root)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def _load_class_label(self, idx):
        return self.data[idx]["class_label"]

    def _load_gt_concept(self, idx):
        item = self.data[idx]
        attr_label = item["uncertain_attribute_label"] if self.uncertain_label else item["attribute_label"]
        return self._subsample(torch.tensor(attr_label, dtype=torch.float32))

    def _load_pseudo_concept(self, idx):
        return self._subsample(self.pseudo_concepts[idx].float())

    def __getitem__(self, idx):
        img = self._load_image(idx)
        class_label = self._load_class_label(idx)

        if self.concept_source == "gt":
            concept_label = self._load_gt_concept(idx)
        else:
            concept_label = self._load_pseudo_concept(idx)

        return img, class_label, concept_label