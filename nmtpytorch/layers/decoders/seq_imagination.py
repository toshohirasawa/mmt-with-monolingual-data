# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn
from ..transformer import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward, get_pad_attn_mask
from ..max_margin import MaxMarginLoss, MaxMarginLoss3D

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, key_size, n_heads, inner_dim, dropout, no_self_attn):
        super(DecoderLayer, self).__init__()

        if no_self_attn:
            self.self_attn = lambda q, k, v, m: [q, None]
        else:
            self.self_attn = MultiHeadAttention(model_dim, n_heads, dropout=dropout)

        self.ctx_attn = MultiHeadAttention(model_dim, n_heads, dropout=dropout)
        self.ff = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, hs, enc_hs, self_attn_mask, ctx_attn_mask, **kwargs):
        h1, self_attn = self.self_attn(hs, hs, hs, self_attn_mask)
        h1 = self.layer_norm(h1 + hs)

        h2, ctx_attn = self.ctx_attn(h1, enc_hs, enc_hs, ctx_attn_mask)
        h2 = self.layer_norm(h2 + h1)

        h3 = self.ff(h2)
        h3 = self.layer_norm(h3 + h2)

        return h3, self_attn, ctx_attn

class SequentialImaginationDecoder(nn.Module):
    """
    Extention of Imagination decoder to predict multiple visual features
    """
    def __init__(self, ctx_dim, feat_dim=2048, n_feats=10,
                key_dim=64, n_heads=8, inner_dim=2048, dropout=0.1,
                margin=0.1, n_negatives=1, **kwargs):
        super().__init__()

        self.ctx_dim = ctx_dim
        self.feat_dim = feat_dim
        self.n_feats = n_feats
        self.margin = margin
        self.n_negatives = n_negatives

        self.pe = PositionalEncoding(model_dim=ctx_dim, max_len=n_feats)

        self.ctx2hid = DecoderLayer(
            model_dim=ctx_dim,
            key_size=key_dim,
            n_heads=n_heads,
            inner_dim=inner_dim,
            dropout=dropout,
            no_self_attn=kwargs.get('no_self_attn', False))

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
        
        hs, _, _ = self.ctx2hid(emb, hs_enc, self_attn_mask, ctx_attn_mask)
        preds = self.hid2out(hs)

        # n_batch x n_feats x model_dim
        return preds

    def forward(self, ctx, feats):

        # calculate loss in each position (i=1, 2, ..., n_feats)
        preds = self.predict(ctx).transpose(0, 1)

        if feats is None:
            return {'pred': preds}

        # select nonzero data
        # assume nonzero batches have complete nonzero feats at any positions
        nonzero_index = feats.nonzero(as_tuple=True)[1].unique().sort()[0]
        if len(nonzero_index) == 0:
            return {'pred': preds}

        nonzero_feats = feats.index_select(dim=1, index=nonzero_index)
        nonzero_preds = preds.index_select(dim=1, index=nonzero_index)

        # loss = sum([self.max_margin_loss(nonzero_out[i], nonzero_feats[i]) for i in range(self.n_feats)])
        # loss = loss / self.n_feats
        loss = self.max_margin_loss(nonzero_preds, nonzero_feats)

        # one for positive sample
        return {
            'loss': loss,
            'pred': preds,
        }