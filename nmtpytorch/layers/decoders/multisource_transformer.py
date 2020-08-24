from copy import deepcopy
import torch
from torch import nn

from ..transformer import (
    PositionalEncoding, 
    MultiHeadAttention, 
    MultiSourceMultiHeadAttention, 
    PositionwiseFeedForward
)
from ..transformer import LabelSmoothingLoss
from ..transformer import get_embs, get_pad_attn_mask

from .transformer import DecoderLayer as TextDecoderLayer

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, feat_dim, feat_ratio, key_size, n_heads, inner_dim, dropout, dropnet):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(model_dim, n_heads, dropout=dropout)
        self.ctx_attn = MultiSourceMultiHeadAttention(model_dim, feat_dim, feat_ratio, n_heads, 
            dropout=dropout, dropnet=dropnet)
        self.ff = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, hs, enc_hs, feat, self_pad_attn_mask, ctx_pad_attn_mask, feat_pad_attn_mask=None):
        h1, self_attn = self.self_attn(hs, hs, hs, self_pad_attn_mask)
        h1 = self.layer_norm(h1 + hs)

        h2, ctx_attn = self.ctx_attn(h1, enc_hs, enc_hs, ctx_pad_attn_mask, 
            feat, feat, feat_pad_attn_mask)
        h2 = self.layer_norm(h2 + h1)

        h3 = self.ff(h2)
        h3 = self.layer_norm(h3 + h2)

        return h3, self_attn, ctx_attn

class MultiSourceTransformerDecoder(nn.Module):
    def __init__(self, n_vocab, model_dim, max_len, layer_types, key_size, n_heads, 
        inner_dim, dropout, dropnet, tied_emb_proj, eps, ctx_name, 
        feat_name, feat_dim, feat_ratio, **kwargs):
        super().__init__()

        self.embs = get_embs(n_vocab, max_len, model_dim, dropout)

        def get_text_layer():
            return TextDecoderLayer(
                model_dim = model_dim, 
                key_size = key_size, 
                n_heads = n_heads, 
                inner_dim = inner_dim,
                dropout = dropout)
        
        def get_fused_layer():
            return DecoderLayer(
                model_dim = model_dim,
                feat_dim = feat_dim,
                feat_ratio = feat_ratio,
                key_size = key_size, 
                n_heads = n_heads, 
                inner_dim = inner_dim,
                dropout = dropout,
                dropnet = dropnet,)

        get_layer = {'t': get_text_layer, 'f': get_fused_layer}

        self.layer_types = layer_types.lower()
        self.layers = nn.ModuleList([get_layer[t]() for t in self.layer_types])

        self.hs2prob = nn.Linear(model_dim, n_vocab)

        # shares embedding and projection weight
        if tied_emb_proj:
            self.hs2prob.weight = self.embs[0].weight
            self.logit_scale = model_dim ** -0.5
        else:
            self.logit_scale = 1.0

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss_fn = LabelSmoothingLoss(eps, n_vocab, padding_index=0)

        self.ctx_name = ctx_name
        self.feat_name = feat_name

    def reset_parameters(self, **kwargs):
        nn.init.normal_(self.embs[0].weight)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.ff.w_1.weight)
            nn.init.xavier_normal_(layer.ff.w_2.weight)
            nn.init.zeros_(layer.ff.w_1.bias)
            nn.init.zeros_(layer.ff.w_2.bias)

    def get_emb(self, idxs, tstep):
        # Only used by search.py
        # idxs: eval_batch_size

        # ret: 1 x batch_size
        return idxs.unsqueeze(0)

    def f_init(self, ctx_dict):
        # Actually initial hidden state is no longer used in Transformer
        # Because transformer decoder needs target outputs so far that search.py does not provide,
        # we pass it by parameter 'h' in f_next function, which is designed to hold rnn state.
        enc_hs, x = ctx_dict[self.ctx_name]
        batch_size = x.size(1)
        return torch.LongTensor(batch_size, 0).to(x.device)

    def f_next(self, ctx_dict, y, h, training=False):
        # ctx_dict: enc_hs (encoder states), x (source sentence)
        # y: batch_size, len_y (default in nmtpytorch)
        # h: None (training), target sentence so far (test)
        y = y.transpose(0, 1)
        if type(h) != type(None): # test
            y = torch.cat([h, y], dim=-1)
        hs = self.embs(y)

        enc_hs, x = ctx_dict[self.ctx_name]
        x = x.transpose(0, 1) # len_x x batch_size
        enc_hs = enc_hs.transpose(0, 1) # len_x x batch_size x model_dim

        #
        feat, feat_mask = ctx_dict[self.feat_name]
        feat = feat.transpose(0, 1)
        feat_mask = feat_mask.transpose(0, 1)
        
        self_pad_attn_mask = get_pad_attn_mask(y, y, masking_seq=True)
        ctx_pad_attn_mask  = get_pad_attn_mask(y, x)
        feat_pad_attn_mask = get_pad_attn_mask(y, feat_mask)

        self_attns, ctx_attns = [], []
        for t, layer in zip(self.layer_types, self.layers):
            hs, self_attn, ctx_attn = layer(
                hs=hs, 
                enc_hs=enc_hs, 
                feat=feat, 
                self_pad_attn_mask=self_pad_attn_mask, 
                ctx_pad_attn_mask=ctx_pad_attn_mask, 
                feat_pad_attn_mask=feat_pad_attn_mask
            )

            self_attns.append(self_attn)
            ctx_attns.append(ctx_attn)

        logit = self.hs2prob(hs)
        if not training:
            logit = logit[:, -1, :]

        logit = logit * self.logit_scale
        log_p = self.log_softmax(logit)

        return log_p, y

    def forward(self, ctx_dict, y):
        # y: len_y x batch_size (default in nmtpytorch)
        # len_y, batch_size= y.size()

        log_p, _ = self.f_next(ctx_dict, y[:-1, :], None, training=True)
        loss = self.loss_fn(log_p, y[1:, :].transpose(0, 1).contiguous().view(-1))

        return {'loss': loss}