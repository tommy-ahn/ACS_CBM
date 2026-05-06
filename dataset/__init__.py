# dataset/__init__.py

from torch.utils.data import DataLoader
from .cub import CUB


def create_dataloader(cfg, task="train"):
    dataset_name = cfg.DATASET.NAME.lower()

    if dataset_name == "cub":
        dataset = CUB(
            cfg=cfg,
            task=task,
            uncertain_label=False,
            selected_concepts=None,
            concept_source=cfg.DATASET.CONCEPT_SOURCE,
            pseudo_dir=cfg.DATASET.PSEUDO_DIR if cfg.DATASET.PSEUDO_DIR != "" else None,
            pseudo_type=cfg.DATASET.PSEUDO_TYPE,
            pseudo_threshold=cfg.DATASET.PSEUDO_THRESHOLD,
            pseudo_topk=cfg.DATASET.PSEUDO_TOPK,
            pseudo_name_path=cfg.DATASET.PSEUDO_NAME_PATH if cfg.DATASET.PSEUDO_NAME_PATH != "" else None,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.DATASET.NAME}")

    if task.lower() == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        num_workers = cfg.TRAIN.NUM_WORKERS
        shuffle = True
    elif task.lower() == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        num_workers = cfg.VAL.NUM_WORKERS
        shuffle = False
    elif task.lower() == "test":
        batch_size = cfg.TEST.BATCH_SIZE
        num_workers = cfg.TEST.NUM_WORKERS
        shuffle = False
    else:
        raise ValueError(f"Unknown task: {task}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    imbalance = getattr(dataset, "imbalance", None)
    return loader, imbalance