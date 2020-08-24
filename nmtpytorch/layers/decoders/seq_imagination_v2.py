# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn
from .transformer import DecoderLayer
from ..transformer import PositionalEncoding, get_pad_attn_mask
from ..max_margin import MaxMarginLoss, MaxMarginLoss3D

class SequentialImaginationDecoderV2(nn.Module):
    """
    Extention of Imagination decoder to predict multiple visual features.
    Changes from V1:
    - Support multiple layers
    """
    def __init__(self, ctx_dim, feat_dim=2048, n_feats=10,
                n_layers=2, key_dim=64, n_heads=8, inner_dim=2048, dropout=0.1,
                margin=0.1, n_negatives=1, **kwargs):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.feat_dim = feat_dim
        self.n_feats = n_feats
        self.margin = margin
        self.n_negatives = n_negatives

        self.pe = PositionalEncoding(model_dim=ctx_dim, max_len=n_feats)

        self.layers = nn.ModuleList([
            DecoderLayer(
                model_dim=ctx_dim,
                key_size=key_dim,
                n_heads=n_heads,
                inner_dim=inner_dim,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        self.hid2out = nn.Linear(self.ctx_dim, self.feat_dim, bias=False)
        self.max_margin_loss = MaxMarginLoss3D(self.margin)

    def reset_parameters(self, **kwargs):
        nn.init.xavier_normal_(self.ctx2hid.ff.w_1.weight)
        nn.init.xavier_normal_(self.ctx2hid.ff.w_2.weight)
        nn.init.zeros_(self.ctx2hid.ff.w_1.bias)
        nn.init.zeros_(self.ctx2hid.ff.w_2.bias)

    def predict(self, ctx):
        hs_enc, x = ctx
        n_feats, n_batch, model_dim = hs_enc.shape

        # n_batch x n_feats x model_dim
        hs_enc = hs_enc.transpose(0, 1)
        x = x.transpose(0, 1) if x is not None else torch.ones(n_batch, hs_enc.shape[1], device=hs_enc.device)

        emb = torch.zeros(n_batch, self.n_feats, self.ctx_dim, device=hs_enc.device)
        emb = self.pe(emb)

        dummy_y = torch.ones(n_batch, self.n_feats, device=hs_enc.device)
        self_attn_mask = get_pad_attn_mask(dummy_y, dummy_y)
        ctx_attn_mask = get_pad_attn_mask(dummy_y, x)
        
        hs = emb
        for layer in self.layers:
            hs, _, _ = layer(hs, hs_enc, self_attn_mask, ctx_attn_mask)

        preds = self.hid2out(hs)

        # n_batch x n_feats x model_dim
        return preds

    def forward(self, ctx, feats):

        # calculate loss in each position (i=1, 2, ..., n_feats)
        preds = self.predict(ctx).transpose(0, 1)

        # select nonzero data
        # assume nonzero batches have complete nonzero feats at any positions
        nonzero_index = feats.nonzero(as_tuple=True)[1].unique().sort()[0]

        nonzero_feats = feats.index_select(dim=1, index=nonzero_index)
        nonzero_preds = preds.index_select(dim=1, index=nonzero_index)

        # loss = sum([self.max_margin_loss(nonzero_out[i], nonzero_feats[i]) for i in range(self.n_feats)])
        # loss = loss / self.n_feats
        loss = self.max_margin_loss(nonzero_preds, nonzero_feats)

        # one for positive sample
        return {
            'loss': loss,
            'preds': preds,
        }