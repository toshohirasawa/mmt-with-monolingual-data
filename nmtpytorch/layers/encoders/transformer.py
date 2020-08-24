from copy import deepcopy
from torch import nn

from ..transformer import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward
from ..transformer import get_embs, get_pad_attn_mask, get_pad_mask

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, key_size, n_heads, inner_dim, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(model_dim, n_heads, dropout=dropout)
        self.ff = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, attn_mask, pad_mask):
        h1, attn = self.attn(x, x, x, attn_mask)
        h1 = self.layer_norm(h1 + x)
        h1 = h1.masked_fill(pad_mask, 0)

        h2 = self.ff(h1)
        h2 = self.layer_norm(h2 + h1)
        h2 = h2.masked_fill(pad_mask, 0)

        return h2, attn

class TransformerEncoder(nn.Module):
    def __init__(self, n_vocab, model_dim, max_len, 
        n_layers, key_size, n_heads, inner_dim, dropout, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.embs = get_embs(n_vocab, max_len, model_dim, dropout)

        layer = EncoderLayer(
            model_dim = model_dim, 
            key_size = key_size, 
            n_heads = n_heads, 
            inner_dim = inner_dim,
            dropout = dropout)
        self.layers = nn.ModuleList([
            deepcopy(layer) for _ in range(n_layers)
        ])

    def reset_parameters(self, **kwargs):
        nn.init.normal_(self.embs[0].weight)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.ff.w_1.weight)
            nn.init.xavier_normal_(layer.ff.w_2.weight)
            nn.init.zeros_(layer.ff.w_1.bias)
            nn.init.zeros_(layer.ff.w_2.bias)

    def forward(self, x, **kwargs):
        # x: len_x x batch_size (default order in nmtpytorch)
        x = x.transpose(0, 1) # batch_size x len_x
        hs = self.embs(x)

        attn_mask = get_pad_attn_mask(x, x)
        pad_mask = get_pad_mask(x)

        attns = []
        for layer in self.layers:
            hs, attn = layer(hs, attn_mask, pad_mask)
            attns.append(attn)
        
        return hs.transpose(0, 1), x.transpose(0, 1), attns

    def get_last_layer(self):
        return self.layers[-1].ff.w_2