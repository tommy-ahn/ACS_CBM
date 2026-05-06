from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar


def create_callbacks(cfg):
    return [
        EarlyStopping(
            monitor="val/loss",
            min_delta=0.0,
            patience=cfg.TRAIN.PATIENCE,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            save_top_k=1,
            monitor="val/avg_cy_acc",
            mode="max",
            filename=f"{cfg.EXP_NAME}_best",
            save_last=True,
        ),
        RichProgressBar(leave=True),
    ]
