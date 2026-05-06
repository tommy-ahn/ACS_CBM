import torch
import torch.nn as nn
from einops import rearrange

from util import compute_accuracy

from .abc import LitABC
from .model_util import create_vision_model


class ConceptEmbeddingModel(LitABC):
    def __init__(self, cfg, imbalance) -> None:
        super().__init__(cfg)
        self.dim = cfg.MODEL.DIM

        self.x2z = create_vision_model(cfg.MODEL.X2Z)
        self.x2z.fc = nn.Identity()
        self.inplanes = self.x2z.inplanes

        self.z2c = nn.Linear(self.inplanes, self.n_concepts * self.dim * 2)
        self.c2p = nn.Linear(self.n_concepts * self.dim * 2, self.n_concepts)
        self.c2y = nn.Linear(self.n_concepts * self.dim, self.n_tasks)

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
        c = self.z2c(z)
        p = self.c2p(c).sigmoid()  # b c

        c = rearrange(c, "b (c d m) -> m b c d", c=self.n_concepts, d=self.dim, m=2)
        pos_c, neg_c = c[0], c[1]  # b c d, b c d

        c = pos_c * p.unsqueeze(-1) + neg_c * (1 - p.unsqueeze(-1))  # b c d

        c = rearrange(c, "b c d -> b (c d)")
        y = self.c2y(c)
        return p, y
