from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .abc import LitABC
from .model_util import create_vision_model
from util import compute_y_accuracy
from .unets.conv import ConvConceptizer

import cv2
import numpy as np


class EnergyConceptBottleneckModel(LitABC):
    def __init__(self, cfg, imbalance=None) -> None:
        super().__init__(cfg)
        self.dim = cfg.MODEL.DIM

        # self.unet = ConvConceptizer(image_size = cfg.DATASET.IMG_SIZE, num_concepts = self.n_concepts, concept_dim = 1, image_channels=3, encoder_channels=(10,),
        #          decoder_channels=(16, 8), kernel_size_conv=5, kernel_size_upsample=(5, 5, 2),
        #          stride_conv=1, stride_pool=2, stride_upsample=(2, 1, 2),
        #          padding_conv=0, padding_upsample=(0, 0, 1))

        self.x2z = create_vision_model(cfg.MODEL.X2Z)
        self.inplanes = self.x2z.inplanes
        self.x2z = torch.nn.Sequential(*(list(self.x2z.children())[:-2]))

        self.dout = cfg.DATASET.IMG_SIZE // 32
        self.proj_z = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=self.n_concepts,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.ReLU(inplace=True),
        )

        cav = torch.empty((self.n_concepts, self.dout * self.dout))
        nn.init.orthogonal_(cav, gain=3)
        self.register_buffer("cav", cav)

        self.z2c = nn.ModuleList(
            [nn.Linear((self.dout * self.dout) * 2, 1) for _ in range(self.n_concepts)]
        )

        self.c2x = Decoder(cfg.DATASET.IMG_SIZE, self.n_concepts, self.dout)

        self.c2y = nn.Linear(self.n_concepts, self.n_tasks)

        # ### Set Loss ###
        self.y_loss_fn = nn.CrossEntropyLoss()
        self.x_loss_fn = nn.MSELoss()

        # self.gradient = None

    def _run_step(self, batch, batch_idx):
        x, y, _ = batch
        pred_c, pred_y, rec_x, grad_list, energy_list = self(*batch)

        y_loss = self.y_loss_fn(pred_y, y) * self.cfg.TRAIN.Y_LOSS_WEIGHT
        x_loss = self.x_loss_fn(rec_x, x) * self.cfg.TRAIN.X_LOSS_WEIGHT
        loss = y_loss + x_loss

        y_accuracy = compute_y_accuracy(pred_y, y)
        outputs = {
            "lr": self.current_lr,
            "loss": loss,
            "x_loss": x_loss,
            "y_loss": y_loss,
            "avg_cy_acc": (y_accuracy + y_accuracy) / 2,
            "c_acc": 0,
            "y_acc": y_accuracy,
        }
        return loss, outputs

    def test(self, batch, batch_idx):
        x, y, c = batch

        pred_c, pred_y, rec_x, grad_list, energy_list = self(*batch)

        rec_x = rec_x.detach().cpu().numpy()
        # x_min, x_max = rec_x.min(), rec_x.max()
        # x_denormalized = (255.0 * (rec_x - x_min) / (x_max - x_min)).astype(np.uint8)
        x_denormalized = (255.0 * rec_x).astype(np.uint8)
        x_denormalized = np.transpose(x_denormalized, (0, 2, 3, 1))

        cv2.imwrite("rec.png", x_denormalized[0])

        x = x.detach().cpu().numpy()
        # x_min, x_max = x.min(), x.max()
        # x_denormalized = (255.0 * (x - x_min) / (x_max - x_min)).astype(np.uint8)
        x_denormalized = (255.0 * x).astype(np.uint8)
        x_denormalized = np.transpose(x_denormalized, (0, 2, 3, 1))

        cv2.imwrite("orig.png", x_denormalized[0])

        grad_to_viz(x, pred_c, grad_list, energy_list, self, self.z2c)

    def forward(self, x, y, c):
        batch_size = x.shape[0]
        # pred_c, rec_x = self.unet(x)
        z = self.x2z(x.cuda())
        z = self.proj_z(z)

        cav = self.cav.unsqueeze(0).repeat(batch_size, 1, 1)
        z = z.view(-1, self.n_concepts, self.dout * self.dout)

        energy_list = []
        for i, m in enumerate(self.z2c):
            _z = z[:, [i], :]
            _c = cav[:, [i], :]
            e = m(torch.cat([_z, _c], -1))
            energy_list.append(e)

        gradient_list = []
        # for i, e in enumerate(energies):
        #     e_grad = torch.autograd.grad([e.sum()], [z])[0][:, i]
        #     e_grad = e_grad.view(-1, self.dout, self.dout)
        #     gradient_list.append(e_grad)

        energies = torch.cat(energy_list, dim=1).relu()
        pred_c = torch.exp(-energies)

        rec_x = self.c2x(energies)

        pred_c = pred_c.squeeze(-1)

        pred_y = self.c2y(pred_c)

        return pred_c, pred_y, rec_x, gradient_list, energy_list

    # def activations_hook(self, grad):
    #     self.gradient = grad

    # @property
    # def get_activations_gradient(self):
    #     return self.gradient


class Decoder(nn.Module):
    def __init__(
        self,
        img_size,
        n_concepts,
        dout,
        dims=[512, 256, 128, 64, 3],
        k=[3, 3, 3, 3, 3],
        s=[2, 2, 2, 2, 2],
        p=[0, 1, 1, 1, 1],
    ):
        super().__init__()
        self.img_size = img_size
        self.n_concepts = n_concepts
        self.dout = dout

        self.dims = [self.n_concepts] + dims

        self.unlinear = nn.Linear(1, self.dout**2)
        self.upsample = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.dims[i],
                        out_channels=self.dims[i + 1],
                        kernel_size=k[i],
                        stride=s[i],
                        padding=p[i],
                    ),
                    nn.ReLU() if i < len(self.dims) - 2 else nn.Identity(),
                )
            )
        # self.upsample.append(nn.Tanh())
        self.upsample.append(nn.Sigmoid())

    def forward(self, c):
        z = self.unlinear(c)
        z = z.view(-1, self.n_concepts, self.dout, self.dout)

        for m in self.upsample:
            z = m(z)

        z = F.interpolate(z, size=(self.img_size, self.img_size), mode="bicubic")
        return z


def grad_to_viz(x, pred_c, grad_list, energy_list, model, z2c):
    x_min, x_max = x.min(), x.max()
    x_denormalized = (255.0 * (x - x_min) / (x_max - x_min)).astype(np.uint8)
    x_denormalized = np.transpose(x_denormalized, (0, 2, 3, 1))

    for i, e in enumerate(energy_list):
        model.zero_grad()
        e[0].backward(retain_graph=True)
        m = z2c[i]
        grads = m.weight.grad[:, :49]
        w = m.weight[:, :49]
        cam = (grads * w).sum(0).relu()

        cam = cam.view(7, 7)
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))

        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        blend = cv2.addWeighted(x_denormalized[0], 0.7, cam_heatmap, 0.3, 0)
        cv2.imwrite("blend.png", blend)
        c_score = pred_c[0][i]

    # for i, grad in enumerate(grad_list):
    #     grad -= grad.min()
    #     grad /= grad.max()

    #     grad = grad.detach().cpu().numpy()[0]
    #     grad = (grad * 255).astype(np.uint8)
    #     grad = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

    #     # Resize gradient image to match the size of x (imagenet normalized tensor)
    #     grad_resized = cv2.resize(grad, (x.shape[3], x.shape[2]))
    #     # grad_resized = grad_resized[..., np.newaxis]
    #     # grad_resized = np.repeat(grad_resized, 3, -1)

    #     # Denormalize x tensor from imagenet normalization

    #     # Blend x (denormalized from imagenet) and grad
    #     blend = cv2.addWeighted(x_denormalized[0], 0.7, grad_resized, 0.3, 0)

    #     c_score = pred_c[0][i]

    #     # Save the blended image
    #     cv2.imwrite("blend.png", blend)
