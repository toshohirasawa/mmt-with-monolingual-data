# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn

class VAGSharedSpaceDecoder(nn.Module):
    """
    Elliott, Kádár - 2017 - Imagination improves Multimodal Translation
    """
    def __init__(self, txt_ctx_size, img_ctx_size, output_size=512, margin=0.1):
        super().__init__()

        self.txt_ctx_size = txt_ctx_size
        self.img_ctx_size = img_ctx_size
        self.output_size = output_size
        self.margin = margin

        self.txt2out = nn.Sequential(
            nn.Linear(self.txt_ctx_size, self.output_size),
            nn.Tanh()
        )
        self.img2out = nn.Sequential(
            nn.Linear(self.img_ctx_size, self.output_size),
            nn.Tanh()
        )

    def forward(self, txt_ctx, img_ctx):

        txt_out = self.txt2out(txt_ctx[0]).squeeze(0)
        img_out = self.img2out(img_ctx[0]).squeeze(0)

        B = txt_out.shape[0]

        # for simplifying cosine distance calculation below
        U_norm = txt_out / txt_out.norm(dim=-1, keepdim=True)

        # extract and normalize feats
        Y_norm = img_out / img_out.norm(dim=-1, keepdim=True)

        # # positive and negative sampling
        # ps_dists = out.mul(feats_).sum(dim=-1, keepdim=True)
        # ns_dists = out.matmul(feats_.t())

        # Implementation from original paper (Max-Margin)
        errors = U_norm.matmul(Y_norm.t())
        diag = errors.diag()
        # all contrastive images for each sentence
        loss_s = self.margin - errors + diag.unsqueeze(-1)
        loss_s = torch.max(loss_s, torch.zeros_like(loss_s))
        # all contrastive sentences for each image
        loss_i = self.margin - errors + diag.unsqueeze(0)
        loss_i = torch.max(loss_i, torch.zeros_like(loss_i))
        # total loss
        loss_tot = loss_s + loss_i
        loss_tot[range(B), range(B)] = 0.0

        # one for positive sample
        return {'loss': loss_tot.mean()}