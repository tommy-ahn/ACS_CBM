import torch
import torch.nn as nn

from util import compute_accuracy

from .abc import LitABC
from .model_util import create_vision_model


class ConceptBottleneckModel(LitABC):
    def __init__(self, cfg, imbalance) -> None:
        super().__init__(cfg)
        self.bool_concept = cfg.MODEL.BOOL_CONCEPTS

        self.x2z = create_vision_model(cfg.MODEL.X2Z)
        self.x2z.fc = nn.Identity()
        self.inplanes = self.x2z.inplanes

        self.z2c = nn.Linear(self.inplanes, self.n_concepts)
        self.c2y = nn.Linear(self.n_concepts, self.n_tasks)

        ### Set Loss ###
        self.c_loss_fn = nn.BCELoss(weight=imbalance)
        self.y_loss_fn = nn.CrossEntropyLoss()

    def _run_step(self, batch, batch_idx):
        _, y, c = batch

        pred_c, pred_y = self(*batch)
        c_loss = self.c_loss_fn(pred_c, c) * self.cfg.TRAIN.C_LOSS_WEIGHT
        y_loss = self.y_loss_fn(pred_y, y) * self.cfg.TRAIN.Y_LOSS_WEIGHT
        loss = c_loss + y_loss

        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            pred_c,
            pred_y,
            c,
            y,
        )

        outputs = {
            "lr": self.current_lr,
            "loss": loss,
            "c_loss": c_loss,
            "y_loss": y_loss,
            "avg_cy_acc": (c_accuracy + y_accuracy) / 2,
            "c_acc": c_accuracy,
            "y_acc": y_accuracy,
        }
        return loss, outputs

    def forward(self, x, y, c):
        z = self.x2z(x)
        pred_c = self.z2c(z).sigmoid()
        if self.bool_concept:
            pred_y = self.c2y((pred_c > 0.5).float())
        else:
            pred_y = self.c2y(pred_c)
        return pred_c, pred_y
