# -*- coding: utf-8 -*-
from collections import defaultdict
import random

import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from .. import FF
from ..attention import get_attention


class VAGConditionalDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 att_type='mlp', dec_init_lambda=0.5,
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False, sent_ctx_name='sent'):
        super().__init__()

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_type = att_type
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.dec_init_lambda = dec_init_lambda
        self.sent_ctx_name = sent_ctx_name

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create attention layer
        Attention = get_attention(self.att_type)
        self.att = Attention(
            self.ctx_size_dict[self.ctx_name],
            self.hidden_size,
            transform_ctx=self.transform_ctx,
            mlp_bias=self.mlp_bias,
            att_activ=self.att_activ,
            att_bottleneck=self.att_bottleneck,
            temp=self.att_temp)

        self.ff_dec_init = FF(self.ctx_size_dict[self.ctx_name], 
            self.hidden_size,
            bias=False, activ='tanh')

        # Create decoders
        self.dec0 = nn.GRUCell(self.input_size, self.hidden_size)
        self.dec1 = nn.GRUCell(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def init_hidden(self, ctx_dict):
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        txt_ctx, txt_ctx_mask = ctx_dict[self.sent_ctx_name]
        mean_ctx = ctx.sum(0).div(ctx_mask.unsqueeze(-1).sum(0)) if ctx_mask is not None else ctx.mean(0)

        mixed_ctx = self.dec_init_lambda * mean_ctx + (1 - self.dec_init_lambda) * txt_ctx

        h_0 = self.ff_dec_init(mixed_ctx)

        return h_0.squeeze(0)

    def get_emb(self, idxs, tstep):
        return self.emb(idxs)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.history = defaultdict(list)
        return self.init_hidden(ctx_dict)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, h)
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        txt_alpha_t, txt_z_t = self.att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])

        if not self.training:
            self.history['alpha_txt'].append(txt_alpha_t)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(txt_z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h2_c2

    def forward(self, ctx_dict, y):
        loss = 0.0

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # Convert token indices to embeddings -> T*B*E
        # Skip <bos> now
        bos = self.get_emb(y[0], 0)
        log_p, h = self.f_next(ctx_dict, bos, h)
        loss += self.nll_loss(log_p, y[1])
        y_emb = self.emb(y[1:])

        for t in range(y_emb.shape[0] - 1):
            emb = y_emb[t]
            log_p, h = self.f_next(ctx_dict, emb, h)
            loss += self.nll_loss(log_p, y[t + 2])

        return {'loss': loss}