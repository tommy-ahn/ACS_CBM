import argparse
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import clip


# ---------------------------------------------------------
# utils
# ---------------------------------------------------------
OLD_CUB_PREFIX = "/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets"


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_concepts(concept_file: str) -> List[str]:
    with open(concept_file, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f.readlines()]
    return [c for c in concepts if len(c) > 0]


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def resolve_img_path(img_path: str, root: str) -> str:
    """
    Try to robustly resolve the actual CUB image path.
    """
    root = os.path.abspath(root)

    # Case 1: old hardcoded prefix
    if img_path.startswith(OLD_CUB_PREFIX):
        img_path = img_path.replace(OLD_CUB_PREFIX, root)

    # Case 2: already valid
    if os.path.exists(img_path):
        return img_path

    # Case 3: relative under root
    candidate = os.path.join(root, img_path)
    if os.path.exists(candidate):
        return candidate

    # Case 4: try root/CUB_200_2011 + tail
    if "CUB_200_2011" in img_path:
        tail = img_path[img_path.find("CUB_200_2011") :]
        candidate = os.path.join(root, tail)
        if os.path.exists(candidate):
            return candidate

    # Case 5: if path contains "images/..."
    if "images/" in img_path.replace("\\", "/"):
        tail = img_path.replace("\\", "/").split("images/", 1)[1]
        candidate = os.path.join(root, "CUB_200_2011", "images", tail)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not resolve image path: {img_path}")


# ---------------------------------------------------------
# dataset for CLIP feature extraction
# ---------------------------------------------------------
class CUBForCLIP(Dataset):
    def __init__(self, root: str, split: str, preprocess):
        super().__init__()
        self.root = root
        self.split = split
        self.preprocess = preprocess

        pkl_path = os.path.join(root, "CUB_200_2011", f"{split}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing split file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = resolve_img_path(item["img_path"], self.root)
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)

        label = item["class_label"] if "class_label" in item else -1
        return image, label


# ---------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------
@torch.no_grad()
def extract_image_features(
    model,
    dataset,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    all_features = []
    all_labels = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    for images, labels in tqdm(loader, desc="Extracting image features"):
        feats = model.encode_image(images.to(device))
        all_features.append(feats.cpu())
        all_labels.append(labels.cpu())

    image_features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return image_features, labels


@torch.no_grad()
def extract_text_features(
    model,
    concepts: List[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    all_features = []

    for i in tqdm(range(0, len(concepts), batch_size), desc="Extracting text features"):
        batch_concepts = concepts[i : i + batch_size]
        tokens = clip.tokenize(batch_concepts).to(device)
        feats = model.encode_text(tokens)
        all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)


# ---------------------------------------------------------
# similarity -> pseudo GT
# ---------------------------------------------------------
@torch.no_grad()
def compute_similarity(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    image_features = normalize_features(image_features.float())
    text_features = normalize_features(text_features.float())
    return image_features @ text_features.T  # [N, C]


@torch.no_grad()
def build_soft_pseudo_gt(
    similarity: torch.Tensor,
    temperature: float = 0.07,
    mode: str = "sigmoid",
) -> torch.Tensor:
    """
    mode:
      - sigmoid : independent soft concept scores
      - minmax  : per-image min-max scaling
      - softmax : per-image distribution over concepts
    """
    mode = mode.lower()

    if mode == "sigmoid":
        return torch.sigmoid(similarity / temperature)

    if mode == "minmax":
        s_min = similarity.min(dim=1, keepdim=True)[0]
        s_max = similarity.max(dim=1, keepdim=True)[0]
        return (similarity - s_min) / (s_max - s_min + 1e-8)

    if mode == "softmax":
        return torch.softmax(similarity / temperature, dim=1)

    raise ValueError(f"Unknown soft mode: {mode}")


@torch.no_grad()
def build_binary_pseudo_gt_threshold(similarity: torch.Tensor, threshold: float) -> torch.Tensor:
    return (similarity >= threshold).float()


@torch.no_grad()
def build_binary_pseudo_gt_topk(similarity: torch.Tensor, topk: int) -> torch.Tensor:
    binary = torch.zeros_like(similarity, dtype=torch.float32)
    idx = torch.topk(similarity, k=topk, dim=1).indices
    binary.scatter_(1, idx, 1.0)
    return binary


@torch.no_grad()
def build_topk_scores(similarity: torch.Tensor, topk: int = 10):
    values, indices = torch.topk(similarity, k=topk, dim=1)
    return values, indices


# ---------------------------------------------------------
# main split pipeline
# ---------------------------------------------------------
def run_one_split(
    root: str,
    split: str,
    concept_file: str,
    out_dir: str,
    clip_name: str,
    batch_size: int,
    num_workers: int,
    device: str,
    threshold: float,
    binary_topk: int,
    score_topk: int,
    soft_mode: str,
    temperature: float,
):
    split_out = os.path.join(out_dir, split)
    make_dir(split_out)

    print(f"\n[INFO] Split: {split}")
    print(f"[INFO] Loading CLIP model: {clip_name}")
    model, preprocess = clip.load(clip_name, device=device)

    print("[INFO] Loading concepts")
    concepts = load_concepts(concept_file)
    print(f"[INFO] #concepts = {len(concepts)}")

    with open(os.path.join(split_out, "concepts.txt"), "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(c + "\n")

    dataset = CUBForCLIP(root=root, split=split, preprocess=preprocess)

    # 1) image features
    image_features, labels = extract_image_features(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    torch.save(image_features, os.path.join(split_out, "image_features.pt"))
    torch.save(labels, os.path.join(split_out, "labels.pt"))

    # 2) text features
    text_features = extract_text_features(
        model=model,
        concepts=concepts,
        batch_size=batch_size,
        device=device,
    )
    torch.save(text_features, os.path.join(split_out, "text_features.pt"))

    # 3) similarity
    similarity = compute_similarity(image_features, text_features)
    torch.save(similarity, os.path.join(split_out, "similarity.pt"))

    # 4) soft pseudo GT
    soft_pseudo_gt = build_soft_pseudo_gt(
        similarity=similarity,
        temperature=temperature,
        mode=soft_mode,
    )
    torch.save(soft_pseudo_gt, os.path.join(split_out, "soft_pseudo_gt.pt"))

    # 5) binary threshold GT
    binary_thresh = build_binary_pseudo_gt_threshold(similarity, threshold=threshold)
    torch.save(
        binary_thresh,
        os.path.join(split_out, f"binary_pseudo_gt_threshold_{threshold:.2f}.pt"),
    )

    # 6) binary top-k GT
    binary_topk_gt = build_binary_pseudo_gt_topk(similarity, topk=binary_topk)
    torch.save(
        binary_topk_gt,
        os.path.join(split_out, f"binary_pseudo_gt_topk_{binary_topk}.pt"),
    )

    # 7) analysis top-k
    topk_values, topk_indices = build_topk_scores(similarity, topk=score_topk)
    torch.save(topk_values, os.path.join(split_out, f"top{score_topk}_scores.pt"))
    torch.save(topk_indices, os.path.join(split_out, f"top{score_topk}_indices.pt"))

    print(f"[INFO] Saved split artifacts to: {split_out}")
    print(f"[INFO] similarity shape: {tuple(similarity.shape)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing CUB_200_2011/")
    parser.add_argument("--concept_file", type=str, required=True, help="Text file with one concept per line")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir")
    parser.add_argument("--clip_name", type=str, default="ViT-B/32")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.28)
    parser.add_argument("--binary_topk", type=int, default=20)
    parser.add_argument("--score_topk", type=int, default=10)
    parser.add_argument("--soft_mode", type=str, default="sigmoid", choices=["sigmoid", "minmax", "softmax"])
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--all_splits", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.all_splits and args.split is None:
        raise ValueError("Use either --split {train|val|test} or --all_splits")

    splits = ["train", "val", "test"] if args.all_splits else [args.split]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    for split in splits:
        run_one_split(
            root=args.root,
            split=split,
            concept_file=args.concept_file,
            out_dir=args.out_dir,
            clip_name=args.clip_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            threshold=args.threshold,
            binary_topk=args.binary_topk,
            score_topk=args.score_topk,
            soft_mode=args.soft_mode,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()