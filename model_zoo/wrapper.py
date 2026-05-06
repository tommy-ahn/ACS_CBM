# import lightning as L
# import torch
# import torch.nn as nn
# from rich import get_console

# from util import MetricLogger, compute_accuracy, make_rich_table


# class LitModelWrapper(L.LightningModule):
#     def __init__(self, cfg, model, imbalance) -> None:
#         super().__init__()
#         self.save_hyperparameters(cfg)
#         self.cfg = cfg
#         self.model = model

#         ### Set Optimizer ###
#         self.optim_name = cfg.TRAIN.OPTIM.NAME
#         self.lr = cfg.TRAIN.OPTIM.LR
#         self.weight_decay = cfg.TRAIN.OPTIM.WD
#         self.momentum = cfg.TRAIN.OPTIM.MOMENTUM
#         self.scheduler_name = cfg.TRAIN.SCHEDULER.NAME

#         ### Set Loss ###
#         self.c_loss_fn = nn.BCELoss(weight=imbalance)
#         self.y_loss_fn = nn.CrossEntropyLoss()

#         ### Set Logger ###
#         self.val_loop_logger = MetricLogger()

#     def _run_step(self, batch, batch_idx):
#         _, y, c = batch

#         pred_c, pred_y = self.model(*batch)
#         c_loss = self.c_loss_fn(pred_c, c) * self.cfg.TRAIN.C_LOSS_WEIGHT
#         y_loss = self.y_loss_fn(pred_y, y) * self.cfg.TRAIN.Y_LOSS_WEIGHT
#         loss = c_loss + y_loss

#         (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
#             pred_c,
#             pred_y,
#             c,
#             y,
#         )

#         outputs = {
#             "lr": self.current_lr,
#             "loss": loss,
#             "c_loss": c_loss,
#             "y_loss": y_loss,
#             "avg_cy_acc": (c_accuracy + y_accuracy) / 2,
#             "c_acc": c_accuracy,
#             "y_acc": y_accuracy,
#         }
#         return loss, outputs

#     def training_step(self, batch, batch_idx):
#         loss, outputs = self._run_step(batch, batch_idx)
#         self.log_dict(
#             outputs,
#             prog_bar=True,
#         )
#         return outputs

#     def validation_step(self, batch, batch_idx):
#         loss, outputs = self._run_step(batch, batch_idx)
#         outputs = {"val/" + k: v for k, v in outputs.items()}
#         self.log_dict(outputs, prog_bar=False, sync_dist=True, on_epoch=True)
#         return outputs

#     def on_validation_epoch_end(self) -> None:
#         super().on_validation_epoch_end()

#         val_results = {
#             k: v for k, v in self._trainer.logged_metrics.items() if "val" in k
#         }

#         self.val_loop_logger.log_dict(val_results)

#         table = make_rich_table(val_results, f"Validation Epoch {self.current_epoch}")
#         table.add_row(
#             "max val/avg_cy_acc",
#             f'{self.val_loop_logger.get_max_value("val/avg_cy_acc"):.5f}',
#         )

#         get_console().print(table)

#     def test_step(self, batch, batch_idx):
#         loss, outputs = self._run_step(batch, batch_idx)
#         outputs = {"test/" + k: v for k, v in outputs.items()}
#         self.log_dict(
#             outputs,
#             prog_bar=False,
#             on_epoch=True,
#             sync_dist=True,
#         )
#         return outputs

#     def configure_optimizers(self):
#         ### Set Optim ###
#         self.optimizer = None
#         params = filter(lambda p: p.requires_grad, self.parameters())
#         if self.optim_name.lower() == "adam":
#             self.optimizer = torch.optim.Adam(
#                 params,
#                 lr=self.lr,
#                 weight_decay=self.weight_decay,
#             )
#         elif self.optim_name.lower() == "adamw":
#             self.optimizer = torch.optim.AdamW(
#                 params,
#                 lr=self.lr,
#                 weight_decay=self.weight_decay,
#             )
#         elif self.optim_name.lower() == "sgd":
#             self.optimizer = torch.optim.SGD(
#                 params,
#                 lr=self.lr,
#                 momentum=self.momentum,
#                 weight_decay=self.weight_decay,
#             )
#         else:
#             raise NotImplementedError(f"optimizer : {self.optim_name} is not supported")

#         ### Set Scheduler ###
#         scheduler = None
#         if self.scheduler_name.lower() == "lrp":
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizer, verbose=True
#             )
#         else:
#             raise NotImplementedError(
#                 f"scheduler : {self.scheduler_name} is not implemented"
#             )

#         return [self.optimizer], [
#             {"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}
#         ]

#     @property
#     def current_lr(self):
#         return self.optimizer.param_groups[0]["lr"]
