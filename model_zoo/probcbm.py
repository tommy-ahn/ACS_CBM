import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .abc import LitABC
from .model_util import create_vision_model


class ProbabilisticConceptBottleneckModel(LitABC):
    def __init__(self, cfg, imbalance) -> None:
        super().__init__(cfg)
        self.dim = cfg.MODEL.DIM
        self.y_dim = cfg.MODEL.Y_DIM
        self.n_samples = cfg.MODEL.N_SAMPLES
        self.eps = 1e-10

        ### Set Trainable Params ###
        self.neg_scale = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
        self.shift = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
        self.y_neg_scale = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
        self.y_mean = nn.Parameter(
            torch.randn(self.n_tasks, self.y_dim), requires_grad=True
        )
        self.cav = nn.Parameter(
            torch.randn(2, self.n_concepts, self.dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.cav, std=1.0 / math.sqrt(self.dim))

        ### Set Modules ###
        self.x2z = create_vision_model(cfg.MODEL.X2Z)
        self.avgpool = self.x2z.avgpool
        self.inplanes = self.x2z.inplanes
        self.x2z = torch.nn.Sequential(*(list(self.x2z.children())[:-1]))

        self.stem = nn.Sequential(
            nn.Conv2d(self.inplanes, self.dim * self.n_concepts, kernel_size=1),
            nn.BatchNorm2d(self.dim * self.n_concepts),
            nn.ReLU(),
        )
        weights_init(self.stem)

        self.mean_head = nn.ModuleList(
            [PIENet(1, self.dim, self.dim, self.dim) for _ in range(self.n_concepts)]
        )
        self.logsigma_head = nn.ModuleList(
            [
                UncertaintyModuleImage(self.dim, self.dim, self.dim)
                for _ in range(self.n_concepts)
            ]
        )

        self.head = nn.Linear(self.dim * self.n_concepts, self.y_dim)
        weights_init(self.head)

    def _run_step(self, batch, batch_idx):
        _, y, c = batch

        pred_c, pred_y = self(*batch)
        raise NotImplementedError

    def forward(self, x, y, c):
        z = self.x2z(x)
        c_mean, c_logsigma = self._z2c(z)
        cav_mean = F.normalize(self.cav, p=2, dim=-1)

        if self.training:
            c_emb = sample_gaussian_tensors(c_mean, c_logsigma, self.n_samples)
        else:
            c_emb = c_mean.unsqueeze(2)

        cav_emb = cav_mean.unsqueeze(-2)
        c_logit, p = self._match_prob(c_emb, cav_emb)
        pred_c = p[..., 1].mean(dim=-1)

        c_uncertainty = self._get_uncertainty(c_logsigma)

        out = {
            "p": pred_c,
            "c_uncertainty": c_uncertainty,
            "c_logit": c_logit,
            "c_emb": c_emb,
            "cav_emb": cav_emb,
            "c_mean": c_mean,
            "c_logsigma": c_logsigma,
            "cav_mean": cav_mean,
            "shift": self.shift,
            "neg_scale": self.neg_scale,
        }

        y_emb = c_emb.permute(0, 2, 1, 3).detach()
        if self.training:  # Intervention
            c = F.one_hot(c.long(), num_classes=2)
            target_c_embs = (
                c.view(*c.shape, 1, 1) * cav_emb.permute(1, 0, 2, 3).unsqueeze(0)
            ).sum(2)
            if target_c_embs.shape[2] != self.n_samples:
                target_c_embs = target_c_embs.repeat(1, 1, self.n_samples, 1)
            target_c_embs = target_c_embs.permute(0, 2, 1, 3).detach()
            # y_emb = torch.where(torch.rand_like(target_c_embs[..., :1, :1]) < self.intervention_prob, target_c_embs, y_emb) # Intervention

        y_emb = y_emb.contiguous().view(*y_emb.shape[:2], -1)
        y_emb = self.head(y_emb)
        dist = torch.sqrt(
            (
                (
                    (
                        y_emb.unsqueeze(1)
                        - self.y_mean.unsqueeze(1)
                        .repeat(1, self.n_samples if self.training else 1, 1)
                        .unsqueeze(0)
                    )
                    ** 2
                ).mean(-1)
                + self.eps
            )
        )
        dist = self.y_neg_scale * dist
        pred_y = F.softmax(-dist, dim=-2).mean(dim=-1)
        if not self.training:
            y_uncertainty = self._get_y_uncertainty(c_logsigma.exp(), self.head.weight)

        return pred_c, pred_y

    def _z2c(self, z):
        z = self.stem(z)
        z_pool = self.avgpool(z)
        z = rearrange(z, "b (c d) h w -> b c (h w) d", c=self.n_concepts, d=self.dim)
        z_pool = z_pool.view(-1, self.n_concepts, self.dim)
        c_mean = torch.stack(
            [self.mean_head[i](z_pool[:, i], z[:, i]) for i in range(self.n_concepts)],
            dim=1,
        )
        c_mean = F.normalize(c_mean, p=2, dim=-1)
        c_logsigma = torch.stack(
            [
                self.logsigma_head[i](z_pool[:, i], z[:, i])
                for i in range(self.n_concepts)
            ],
            dim=1,
        )
        c_logsigma = torch.clip(c_logsigma, max=10)
        return c_mean, c_logsigma

    def _match_prob(self, c_emb, cav_emb, reduction=False):
        dist = batchwise_cdist(c_emb, cav_emb)
        dist = dist.permute(0, 2, 3, 1)

        logits = -self.neg_scale.view(1, -1, 1, 1) * dist
        prob = F.softmax(logits, dim=-1)
        if reduction:
            return logits.mean(axis=-2), prob.mean(axis=-2)
        else:
            return logits, prob

    def _get_uncertainty(self, c_logsigma):
        uncertainty = c_logsigma.mean(dim=-1).exp()
        return uncertainty

    def _get_y_uncertainty(self, c_logsigma, weight):
        all_logsigma = c_logsigma.view(c_logsigma.shape[0], -1)
        cov = all_logsigma
        cov = torch.eye(cov.shape[1]).unsqueeze(0).to(cov.device) * cov.unsqueeze(-1)
        full_cov = F.linear(F.linear(cov, weight).transpose(1, 2), weight)
        _, s, _ = torch.linalg.svd(full_cov)
        c = (s + self.eps).log()
        return c.mean(dim=1).exp()


def weights_init(module):
    """Initialize the weights"""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def sample_gaussian_tensors(mu, logsigma, n_samples):
    eps = torch.randn(
        mu.size(0),
        mu.size(1),
        n_samples,
        mu.size(2),
        dtype=mu.dtype,
        device=mu.device,
    )
    samples_sigma = eps.mul(torch.exp(logsigma.unsqueeze(2) * 0.5))
    samples = samples_sigma.add_(mu.unsqueeze(2))
    return samples


def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)
    The following broadcasting operation will be computed:
    (N x Nc x 1 x K x D) - (N x Nc x K x 1 x D) = (N x Nc x K x K x D)
    Parameters
    ----------
    samples1: torch.Tensor (shape: N x Nc x K x D)
    samples2: torch.Tensor (shape: N x Nc x K x D)
    Returns
    -------
    batchwise distance: N x Nc x K ** 2
    """
    if len(samples1.size()) not in [3, 4, 5] or len(samples2.size()) not in [3, 4, 5]:
        raise RuntimeError(
            "expected: 4-dim tensors, got: {}, {}".format(
                samples1.size(), samples2.size()
            )
        )

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    elif samples1.shape[1] == samples2.shape[1]:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(0)
        result = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
        return result.view(*result.shape[:-2], -1)
    else:
        raise RuntimeError(
            f"samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities "
            "are non-broadcastable."
        )
    if len(samples1.size()) == 5:
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps)
    elif len(samples1.size()) == 4:
        samples1 = samples1.unsqueeze(2)
        samples2 = samples2.unsqueeze(3)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(
            batch_size, samples1.size(1), -1
        )
    else:
        samples1 = samples1.unsqueeze(1)
        samples2 = samples2.unsqueeze(2)
        return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(
            batch_size, -1
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_in, d_hidden):
        super().__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    def __init__(self, n_embs, d_in, d_out, d_h, dropout=0.0) -> None:
        super().__init__()

        self.n_embs = n_embs
        self.f_fc = nn.Linear(d_in, d_out)

        self.attention = MultiHeadSelfAttention(n_embs, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.f_fc.weight)
        nn.init.constant_(self.f_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        out = self.f_fc(out)
        if self.n_embs > 1:
            out = out.unsqueeze(1).repeat(1, self.n_embs, 1)
        out = self.layer_norm(out + residual)
        return out


class UncertaintyModuleImage(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return out
