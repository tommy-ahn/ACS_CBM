from abc import *

import lightning as L
import torch
from rich import get_console

from util import MetricLogger, make_rich_table


class LitABC(L.LightningModule, metaclass=ABCMeta):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.n_tasks = cfg.DATASET.N_TASKS
        self.n_concepts = cfg.DATASET.N_CONCEPTS

        ### Set Optimizer ###
        self.optim_name = cfg.TRAIN.OPTIM.NAME
        self.lr = cfg.TRAIN.OPTIM.LR
        self.weight_decay = cfg.TRAIN.OPTIM.WD
        self.momentum = cfg.TRAIN.OPTIM.MOMENTUM
        self.scheduler_name = cfg.TRAIN.SCHEDULER.NAME

        ### Set Logger ###
        self.val_loop_logger = MetricLogger()

    @abstractmethod
    def _run_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss, outputs = self._run_step(batch, batch_idx)
        self.log_dict(
            outputs,
            prog_bar=True,
        )
        return outputs

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._run_step(batch, batch_idx)
        outputs = {"val/" + k: v for k, v in outputs.items()}
        self.log_dict(outputs, prog_bar=False, sync_dist=True, on_epoch=True)
        return outputs

    def test_step(self, batch, batch_idx):
        loss, outputs = self._run_step(batch, batch_idx)
        outputs = {"test/" + k: v for k, v in outputs.items()}
        self.log_dict(
            outputs,
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
        )
        return outputs

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        val_results = {
            k: v for k, v in self._trainer.logged_metrics.items() if "val" in k
        }

        self.val_loop_logger.log_dict(val_results)

        table = make_rich_table(val_results, f"Validation Epoch {self.current_epoch}")
        table.add_row(
            "max val/avg_cy_acc",
            f'{self.val_loop_logger.get_max_value("val/avg_cy_acc"):.5f}',
        )

        get_console().print(table)

    def configure_optimizers(self):
        ### Set Optim ###
        optimizer = None
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optim_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optim_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optim_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError(f"optimizer : {self.optim_name} is not supported")

        ### Set Scheduler ###
        scheduler = None
        if self.scheduler_name.lower() == "lrp":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, verbose=True
            )
        else:
            raise NotImplementedError(
                f"scheduler : {self.scheduler_name} is not implemented"
            )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}
        ]

    @property
    def current_lr(self):
        return self.optimizers().param_groups[0]["lr"]
        # return self.optimizer.param_groups[0]["lr"]
