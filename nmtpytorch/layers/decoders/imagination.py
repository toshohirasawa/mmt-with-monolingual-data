# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


from ...utils.nn import get_activation_fn
from ..max_margin import MaxMarginLoss

class ImaginationDecoder(nn.Module):
    """
    Elliott, Kádár - 2017 - Imagination improves Multimodal Translation
    """
    def __init__(self, ctx_dim, feat_dim=2048, att_activ='tanh', margin=0.1, pool_type='last'):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.feat_dim = feat_dim
        self.margin = margin

        self.hid2out = nn.Linear(self.ctx_dim, self.feat_dim, bias=False)
        self.activ = get_activation_fn(att_activ)
        self.pooling = {
            'mean': self.mean_pooling,
            'max' : self.max_pooling,
            'last': self.last_pooling,
        }[pool_type]
        self.max_margin_loss = MaxMarginLoss(margin)
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

    def mean_pooling(self, ctx):
        ctx_, mask = ctx

        # if batch is fullly on board, the mask given is None
        if mask is None:
            mask = torch.ones(ctx_.size(0), ctx_.size(1), device=ctx_.device)

        n_tokens = mask.sum(dim=0).type(ctx_.dtype)
        mean_ctx = ctx_.sum(dim=0) / n_tokens.unsqueeze(-1)

        return mean_ctx

    def max_pooling(self, ctx):
        ctx_, mask = ctx
        # if batch is fullly on board, the mask given is None
        if mask is None:
            mask = torch.ones(ctx_.size(0), ctx_.size(1), device=ctx_.device)
        # to avoid selecting elements in paddings, set -inf to them
        ctx_ = ctx_.masked_fill(mask.unsqueeze(-1) == 0., -float('inf'))
        max_ctx, _ = ctx_.max(dim=0)

        return max_ctx
    
    def last_pooling(self, ctx):
        '''retrieve the hidden state corresponding to <eos>
        '''
        ctx_, mask = ctx
        # if batch is fullly on board, the mask given is None
        if mask is None:
            mask = torch.ones(ctx_.size(0), ctx_.size(1), device=ctx_.device)
        last_idx = mask.sum(dim=0) - 1
        last_ctx = ctx_[last_idx, torch.arange(ctx_.size(1)), :]
        return last_ctx

    def forward(self, ctx, feats):
        # predict image feats (size: TxBxS)
        ctx, mask = ctx
        feats, _  = feats

        if mask is not None:
            n_tokens = mask.sum(dim=0).type(ctx.dtype)
            mean_ctx = ctx.sum(dim=0) / n_tokens.unsqueeze(-1)
        else:
            mean_ctx = ctx.mean(dim=0)

        out = self.hid2out(mean_ctx)
        out = self.activ(out)

        return {'loss': self.max_margin_loss(out, feats.squeeze())}
