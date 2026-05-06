import argparse

import lightning as L
from lightning.pytorch import loggers
import torch

from configs import cfg
from dataset import create_dataloader
from model_zoo import create_model
from model_zoo.callbacks import create_callbacks


def main(cfg):
    ### Clean Up ###
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L.seed_everything(cfg.SEED)

    ### Create Dataloader ###
    train_loader, imbalance = create_dataloader(cfg, task="train")
    val_loader, _ = create_dataloader(cfg, task="val")
    test_loader, _ = create_dataloader(cfg, task="test")

    ### Create Model ###
    model = create_model(cfg, imbalance)
    lit_callbacks = create_callbacks(cfg)

    csv_logger = loggers.CSVLogger(save_dir=cfg.LOG_DIR, name=cfg.EXP_NAME)

    ### Set PL Trainer ###
    trainer = L.Trainer(
        max_epochs=cfg.TRAIN.EPOCHS,
        devices=[0],
        logger=csv_logger,
        callbacks=lit_callbacks,
        check_val_every_n_epoch=cfg.VAL.FREQUENCY,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, test_loader, verbose=True)
    print(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/grace_cbm.yml",
        type=str,
        help="A yaml file for overriding parameters sepcification in this module.",
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    main(cfg)
