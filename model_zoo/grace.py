from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abc import LitABC
from .model_util import create_vision_model


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)


def concept_acc(pred_c: torch.Tensor, c: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (pred_c >= threshold).float()
    tgt_bin = (c >= threshold).float()
    return (pred_bin == tgt_bin).float().mean()


def concept_mae(pred_c: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (pred_c - c).abs().mean()


def masked_concept_acc(
    pred_c: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    pred_bin = (pred_c >= threshold).float()
    tgt_bin = (c >= threshold).float()
    correct = ((pred_bin == tgt_bin).float() * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return correct / denom


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraceConceptBottleneckModel(LitABC):
    """
    3-stage RL-main CBM

    Stage 1: plain CBM pretraining
    Stage 2: freeze backbone, train actor/critic only with RL
    Stage 3: joint fine-tune with full-path anchor + weak selected-path supervision
    """

    def __init__(self, cfg, imbalance):
        super().__init__(cfg)
        self.cfg = cfg

        self.n_concepts = int(cfg.DATASET.N_CONCEPTS)
        self.n_tasks = int(cfg.DATASET.N_TASKS)
        self.bool_concept = bool(getattr(cfg.MODEL, "BOOL_CONCEPTS", False))

        # -------------------------
        # backbone
        # -------------------------
        self.x2z = create_vision_model(cfg.MODEL.X2Z)
        self.x2z.fc = nn.Identity()
        self.feat_dim = self.x2z.inplanes

        self.z2c = nn.Linear(self.feat_dim, self.n_concepts)
        self.c2y = nn.Linear(self.n_concepts, self.n_tasks)

        # -------------------------
        # actor / critic
        # -------------------------
        actor_hidden = int(getattr(cfg.GRACE, "ACTOR_HIDDEN_DIM", 512))
        critic_hidden = int(getattr(cfg.GRACE, "CRITIC_HIDDEN_DIM", 256))
        actor_dropout = float(getattr(cfg.GRACE, "ACTOR_DROPOUT", 0.0))
        critic_dropout = float(getattr(cfg.GRACE, "CRITIC_DROPOUT", 0.0))

        self.actor = MLP(self.feat_dim, actor_hidden, self.n_concepts, actor_dropout)
        self.critic = MLP(self.feat_dim, critic_hidden, 1, critic_dropout)

        # -------------------------
        # losses
        # -------------------------
        if imbalance is not None:
            self.c_loss_fn = nn.BCELoss(weight=imbalance)
        else:
            self.c_loss_fn = nn.BCELoss()

        self.c_loss_fn_none = nn.BCELoss(reduction="none")
        self.y_loss_fn = nn.CrossEntropyLoss()
        self.y_loss_fn_none = nn.CrossEntropyLoss(reduction="none")

        # -------------------------
        # weights from config (previous code ignored these)
        # -------------------------
        self.c_loss_weight = float(getattr(cfg.TRAIN, "C_LOSS_WEIGHT", 1.0))
        self.y_loss_weight = float(getattr(cfg.TRAIN, "Y_LOSS_WEIGHT", 1.0))

        # -------------------------
        # hyperparams
        # -------------------------
        self.K = int(cfg.GRACE.K)

        self.stage1_epochs = int(getattr(cfg.GRACE, "STAGE1_EPOCHS", 80))
        self.stage2_epochs = int(getattr(cfg.GRACE, "STAGE2_EPOCHS", 40))
        self.stage3_epochs = int(getattr(cfg.GRACE, "STAGE3_EPOCHS", 30))

        self.lambda_all = float(getattr(cfg.GRACE, "FULL_Y_WEIGHT", 1.0))
        self.lambda_sel = float(getattr(cfg.GRACE, "SEL_Y_WEIGHT", 0.2))

        self.entropy_coef_start = float(getattr(cfg.GRACE, "ENTROPY_COEF", 5e-2))
        self.entropy_coef_end = float(getattr(cfg.GRACE, "ENTROPY_COEF_END", 5e-3))
        self.entropy_anneal_epochs = int(getattr(cfg.GRACE, "ENTROPY_ANNEAL_EPOCHS", 50))

        self.value_coef = float(getattr(cfg.GRACE, "VALUE_COEF", 0.5))
        self.adv_eps = float(getattr(cfg.GRACE, "ADV_EPS", 1e-6))
        self.reward_ema_momentum = float(getattr(cfg.GRACE, "REWARD_EMA_MOMENTUM", 0.95))
        self.max_grad_norm = float(getattr(cfg.GRACE, "MAX_GRAD_NORM", 5.0))
        self.concept_acc_threshold = float(getattr(cfg.GRACE, "CONCEPT_ACC_THRESHOLD", 0.5))

        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_std", torch.tensor(1.0))

        self.automatic_optimization = False

    # =====================================================
    # helpers
    # =====================================================
    def get_stage(self) -> int:
        if self.current_epoch < self.stage1_epochs:
            return 1
        elif self.current_epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        else:
            return 3

    def current_entropy_coef(self) -> float:
        if self.entropy_anneal_epochs <= 0:
            return self.entropy_coef_end
        t = min(float(self.current_epoch) / float(self.entropy_anneal_epochs), 1.0)
        return (1.0 - t) * self.entropy_coef_start + t * self.entropy_coef_end

    def forward(self, x: torch.Tensor):
        z = self.x2z(x)
        pred_c = torch.sigmoid(self.z2c(z))
        pred_y = self.classify_from_concepts(pred_c, None)
        return z, pred_c, pred_y

    def classify_from_concepts(self, pred_c: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            used_c = pred_c
        else:
            used_c = pred_c * mask

        if self.bool_concept:
            used_c = (used_c > 0.5).float()

        return self.c2y(used_c)

    def freeze_backbone(self):
        for p in self.x2z.parameters():
            p.requires_grad_(False)
        for p in self.z2c.parameters():
            p.requires_grad_(False)
        for p in self.c2y.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.x2z.parameters():
            p.requires_grad_(True)
        for p in self.z2c.parameters():
            p.requires_grad_(True)
        for p in self.c2y.parameters():
            p.requires_grad_(True)

    def sample_subset_pl(self, logits: torch.Tensor, k: int):
        """
        Plackett-Luce exact-k subset sampling.
        returns:
            idx: [B, K]
            mask: [B, C]
            logprob: [B]
        """
        B, C = logits.shape
        device = logits.device

        remaining = torch.ones((B, C), dtype=torch.bool, device=device)
        chosen = []
        total_logprob = torch.zeros(B, device=device)

        for _ in range(k):
            masked_logits = logits.masked_fill(~remaining, -1e9)
            probs = F.softmax(masked_logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)

            log_probs = F.log_softmax(masked_logits, dim=-1)
            total_logprob += log_probs[torch.arange(B, device=device), idx]

            chosen.append(idx)
            remaining[torch.arange(B, device=device), idx] = False

        idx = torch.stack(chosen, dim=1)
        mask = torch.zeros_like(logits, dtype=torch.float32)
        mask.scatter_(1, idx, 1.0)

        return idx, mask, total_logprob

    def greedy_subset(self, logits: torch.Tensor, k: int):
        idx = torch.topk(logits, k=k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.float32)
        mask.scatter_(1, idx, 1.0)
        return idx, mask

    def configure_optimizers(self):
        wd = float(getattr(self.cfg.TRAIN.OPTIM, "WD", 0.0))
        opt_name = str(getattr(self.cfg.TRAIN.OPTIM, "NAME", "adam")).lower()
        backbone_lr = float(getattr(self.cfg.TRAIN.OPTIM, "LR", 1e-3))
        momentum = float(getattr(self.cfg.TRAIN.OPTIM, "MOMENTUM", 0.9))

        actor_lr = float(getattr(self.cfg.GRACE, "ACTOR_LR", 3e-5))
        critic_lr = float(getattr(self.cfg.GRACE, "CRITIC_LR", 1e-4))

        backbone_params = (
            list(self.x2z.parameters()) +
            list(self.z2c.parameters()) +
            list(self.c2y.parameters())
        )

        # fix: respect config optimizer instead of always using Adam
        if opt_name == "sgd":
            opt_backbone = torch.optim.SGD(
                backbone_params,
                lr=backbone_lr,
                momentum=momentum,
                weight_decay=wd,
                nesterov=False,
            )
        elif opt_name == "adamw":
            opt_backbone = torch.optim.AdamW(
                backbone_params,
                lr=min(backbone_lr, 1e-3),
                weight_decay=wd,
            )
        else:
            opt_backbone = torch.optim.Adam(
                backbone_params,
                lr=min(backbone_lr, 1e-3),
                weight_decay=wd,
            )

        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=wd)
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=wd)

        return [opt_backbone, opt_actor, opt_critic]

    # =====================================================
    # reward
    # =====================================================
    def compute_task_gap_reward(
        self,
        pred_y_all: torch.Tensor,
        pred_y_sel: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        reward = CE(all) - CE(sel)
        bigger is better
        """
        y_loss_all = self.y_loss_fn_none(pred_y_all, y)
        y_loss_sel = self.y_loss_fn_none(pred_y_sel, y)
        reward = y_loss_all - y_loss_sel
        return reward, y_loss_all, y_loss_sel

    # =====================================================
    # main step
    # =====================================================
    def _run_step(self, batch, batch_idx):
        x, y, c = batch
        stage = self.get_stage()

        # freeze / unfreeze before forward
        if stage == 2:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        z, pred_c, pred_y_all = self.forward(x)

        logits_det = self.actor(z.detach())
        _, mask_det = self.greedy_subset(logits_det, self.K)
        pred_y_sel_det = self.classify_from_concepts(pred_c, mask_det)

        loss_c = self.c_loss_fn(pred_c, c) * self.c_loss_weight
        loss_y_all = self.y_loss_fn(pred_y_all, y) * self.y_loss_weight
        loss_y_sel = self.y_loss_fn(pred_y_sel_det, y) * self.y_loss_weight

        # =====================================================
        # validation / test
        # =====================================================
        if not self.training:
            with torch.no_grad():
                if stage == 1:
                    loss = loss_c + loss_y_all
                else:
                    loss = loss_c + self.lambda_all * loss_y_all + self.lambda_sel * loss_y_sel

                y_acc_all = (pred_y_all.argmax(dim=1) == y).float().mean()
                y_acc_sel = (pred_y_sel_det.argmax(dim=1) == y).float().mean()
                c_acc_val = concept_acc(pred_c, c, self.concept_acc_threshold)
                c_mae_val = concept_mae(pred_c, c)
                selected_c_acc_val = masked_concept_acc(pred_c, c, mask_det, self.concept_acc_threshold)

            return loss, {
                "loss": loss.detach(),
                "avg_cy_acc": y_acc_sel.detach(),
                "y_acc_all": y_acc_all.detach(),
                "y_acc_sel": y_acc_sel.detach(),
                "c_acc": c_acc_val.detach(),
                "c_mae": c_mae_val.detach(),
                "selected_c_acc": selected_c_acc_val.detach(),
                "stage": torch.tensor(float(stage), device=self.device),
            }

        opt_backbone, opt_actor, opt_critic = self.optimizers()

        # -------------------------
        # Stage 1: plain CBM pretraining
        # -------------------------
        if stage == 1:
            loss = loss_c + loss_y_all

            opt_backbone.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                list(self.x2z.parameters()) +
                list(self.z2c.parameters()) +
                list(self.c2y.parameters()),
                self.max_grad_norm,
            )
            opt_backbone.step()

            with torch.no_grad():
                y_acc_all = (pred_y_all.argmax(dim=1) == y).float().mean()
                y_acc_sel = (pred_y_sel_det.argmax(dim=1) == y).float().mean()
                c_acc_val = concept_acc(pred_c, c, self.concept_acc_threshold)
                c_mae_val = concept_mae(pred_c, c)
                selected_c_acc_val = masked_concept_acc(pred_c, c, mask_det, self.concept_acc_threshold)

            return loss.detach(), {
                "loss": loss.detach(),
                "y_acc_all": y_acc_all.detach(),
                "y_acc_sel": y_acc_sel.detach(),
                "c_acc": c_acc_val.detach(),
                "c_mae": c_mae_val.detach(),
                "selected_c_acc": selected_c_acc_val.detach(),
                "stage": torch.tensor(1.0, device=self.device),
            }

        # -------------------------
        # Stage 2: freeze backbone, RL only
        # -------------------------
        if stage == 2:
            logits_sample = self.actor(z.detach())
            _, mask_sample, logprob = self.sample_subset_pl(logits_sample, self.K)

            with torch.no_grad():
                pred_c_rl = pred_c.detach()
                pred_y_all_rl = self.classify_from_concepts(pred_c_rl, None)
                pred_y_sel_rl = self.classify_from_concepts(pred_c_rl, mask_sample)

                reward, y_loss_all_rl, y_loss_sel_rl = self.compute_task_gap_reward(
                    pred_y_all_rl, pred_y_sel_rl, y
                )

                self.reward_mean.mul_(self.reward_ema_momentum).add_(
                    (1.0 - self.reward_ema_momentum) * reward.mean()
                )
                self.reward_std.mul_(self.reward_ema_momentum).add_(
                    (1.0 - self.reward_ema_momentum) * reward.std(unbiased=False).clamp_min(1e-6)
                )

                reward_norm = (reward - self.reward_mean) / (self.reward_std + self.adv_eps)

            value = self.critic(z.detach()).squeeze(-1)
            advantage = reward_norm - value

            entropy_coef = self.current_entropy_coef()
            actor_loss = -(advantage.detach() * logprob).mean() - entropy_coef * entropy_from_logits(logits_sample).mean()
            critic_loss = F.mse_loss(value, reward_norm.detach())
            rl_loss = actor_loss + self.value_coef * critic_loss

            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            self.manual_backward(rl_loss)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            opt_actor.step()
            opt_critic.step()

            with torch.no_grad():
                y_acc_all = (pred_y_all.argmax(dim=1) == y).float().mean()
                y_acc_sel = (pred_y_sel_det.argmax(dim=1) == y).float().mean()
                c_acc_val = concept_acc(pred_c, c, self.concept_acc_threshold)
                c_mae_val = concept_mae(pred_c, c)
                selected_c_acc_val = masked_concept_acc(pred_c, c, mask_det, self.concept_acc_threshold)

            log_loss = loss_c + loss_y_all

            return log_loss.detach(), {
                "loss": log_loss.detach(),
                "reward": reward.mean().detach(),
                "reward_std": reward.std(unbiased=False).detach(),
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                "entropy": entropy_from_logits(logits_sample).mean().detach(),
                "entropy_coef": torch.tensor(float(entropy_coef), device=self.device),
                "gap_y_loss_all": y_loss_all_rl.mean().detach(),
                "gap_y_loss_sel": y_loss_sel_rl.mean().detach(),
                "y_acc_all": y_acc_all.detach(),
                "y_acc_sel": y_acc_sel.detach(),
                "c_acc": c_acc_val.detach(),
                "c_mae": c_mae_val.detach(),
                "selected_c_acc": selected_c_acc_val.detach(),
                "stage": torch.tensor(2.0, device=self.device),
            }

        # -------------------------
        # Stage 3: joint fine-tuning
        # -------------------------
        logits_sample = self.actor(z.detach())
        _, mask_sample, logprob = self.sample_subset_pl(logits_sample, self.K)

        with torch.no_grad():
            pred_c_rl = pred_c.detach()
            pred_y_all_rl = self.classify_from_concepts(pred_c_rl, None)
            pred_y_sel_rl = self.classify_from_concepts(pred_c_rl, mask_sample)

            reward, y_loss_all_rl, y_loss_sel_rl = self.compute_task_gap_reward(
                pred_y_all_rl, pred_y_sel_rl, y
            )

            self.reward_mean.mul_(self.reward_ema_momentum).add_(
                (1.0 - self.reward_ema_momentum) * reward.mean()
            )
            self.reward_std.mul_(self.reward_ema_momentum).add_(
                (1.0 - self.reward_ema_momentum) * reward.std(unbiased=False).clamp_min(1e-6)
            )

            reward_norm = (reward - self.reward_mean) / (self.reward_std + self.adv_eps)

        value = self.critic(z.detach()).squeeze(-1)
        advantage = reward_norm - value

        entropy_coef = self.current_entropy_coef()
        actor_loss = -(advantage.detach() * logprob).mean() - entropy_coef * entropy_from_logits(logits_sample).mean()
        critic_loss = F.mse_loss(value, reward_norm.detach())
        rl_loss = actor_loss + self.value_coef * critic_loss

        opt_actor.zero_grad(set_to_none=True)
        opt_critic.zero_grad(set_to_none=True)
        self.manual_backward(rl_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        opt_actor.step()
        opt_critic.step()

        backbone_loss = loss_c + self.lambda_all * loss_y_all + self.lambda_sel * loss_y_sel

        opt_backbone.zero_grad(set_to_none=True)
        self.manual_backward(backbone_loss)
        torch.nn.utils.clip_grad_norm_(
            list(self.x2z.parameters()) +
            list(self.z2c.parameters()) +
            list(self.c2y.parameters()),
            self.max_grad_norm,
        )
        opt_backbone.step()

        with torch.no_grad():
            y_acc_all = (pred_y_all.argmax(dim=1) == y).float().mean()
            y_acc_sel = (pred_y_sel_det.argmax(dim=1) == y).float().mean()
            c_acc_val = concept_acc(pred_c, c, self.concept_acc_threshold)
            c_mae_val = concept_mae(pred_c, c)
            selected_c_acc_val = masked_concept_acc(pred_c, c, mask_det, self.concept_acc_threshold)

        return backbone_loss.detach(), {
            "loss": backbone_loss.detach(),
            "reward": reward.mean().detach(),
            "reward_std": reward.std(unbiased=False).detach(),
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy_from_logits(logits_sample).mean().detach(),
            "entropy_coef": torch.tensor(float(entropy_coef), device=self.device),
            "gap_y_loss_all": y_loss_all_rl.mean().detach(),
            "gap_y_loss_sel": y_loss_sel_rl.mean().detach(),
            "y_acc_all": y_acc_all.detach(),
            "y_acc_sel": y_acc_sel.detach(),
            "c_acc": c_acc_val.detach(),
            "c_mae": c_mae_val.detach(),
            "selected_c_acc": selected_c_acc_val.detach(),
            "stage": torch.tensor(3.0, device=self.device),
        }