from copy import deepcopy
import torch
from torch import nn

from ..transformer import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward
from ..transformer import LabelSmoothingLoss
from ..transformer import get_embs, get_pad_attn_mask

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, key_size, n_heads, inner_dim, dropout):
        super(DecoderLayer, self).__init__()

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

class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab, model_dim, max_len, n_layers, key_size, n_heads, 
        inner_dim, dropout, tied_emb_proj, eps, ctx_name, **kwargs):
        super(TransformerDecoder, self).__init__()

        self.embs = get_embs(n_vocab, max_len, model_dim, dropout)

        layer = DecoderLayer(
            model_dim = model_dim, 
            key_size = key_size, 
            n_heads = n_heads, 
            inner_dim = inner_dim, 
            dropout = dropout)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

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

        self_attn_mask = get_pad_attn_mask(y, y, masking_seq=True)
        ctx_attn_mask = get_pad_attn_mask(y, x)
        self_attns, ctx_attns = [], []
        for layer in self.layers:
            hs, self_attn, ctx_attn = layer(hs, enc_hs, self_attn_mask, ctx_attn_mask)
            self_attns.append(self_attn)
            ctx_attns.append(ctx_attn)

        logit = self.hs2prob(hs)
        if not training:
            logit = logit[:, -1, :]
        logit = logit * self.logit_scale
        log_p = self.log_softmax(logit)

        # ret_1: batch_size x n_vocab
        # ret_2: batch_size x len
        return log_p, y

    def forward(self, ctx_dict, y, **kwargs):
        # y: len_y x batch_size (default in nmtpytorch)
        # len_y, batch_size= y.size()

        log_p, _ = self.f_next(ctx_dict, y[:-1, :], None, training=True)
        loss = self.loss_fn(log_p, y[1:, :].transpose(0, 1).contiguous().view(-1))

        return {'loss': loss}
