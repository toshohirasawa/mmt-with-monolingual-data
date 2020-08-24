import logging

import torch
from torch import nn
import numpy as np
import math
import regex as re

from ..layers import TransformerEncoder, MultiSourceTransformerDecoder
from .transformer import Transformer
from ..utils.misc import get_n_params, get_feat_mask
from ..utils.data import sort_batch
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')

class AttentiveMultimodalTransformer(Transformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'dec_layer_types': 'ffffff',
            'feat_ratio': 0.5,
            'dropnet': 0.0,
            'feat_name': 'feats',
            'feat_dim': 2048,
            'n_feats': 10,
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.feat_name = self.opts.model['feat_name']

    def reset_parameters(self):
        super().reset_parameters()

    def update_state_dict(self, state_dict, model_type):
        if model_type == 'Transformer':
            # dec.layers.[0-9]+.ctx_attn. => dec.layers.[0-9]+.ctx_attn.ctx_attn.
            cov_keys = []
            for key, value in state_dict.items():
                m = re.search('(dec\.layers\.[0-9]+\.ctx_attn)\.(.+)', key)
                if m is not None:
                    new_key = m[1] + ".ctx_attn." + m[2]
                    cov_keys.append([key, new_key])

            for key, new_key in cov_keys:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        return state_dict

    def setup(self, is_train=True):
        self.enc = TransformerEncoder(
            n_vocab = self.n_src_vocab,
            model_dim = self.opts.model['model_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['max_len'],
            dropout = self.opts.model['dropout'],
        )
        self.dec = MultiSourceTransformerDecoder(
            n_vocab = self.n_trg_vocab,
            model_dim = self.opts.model['model_dim'],
            n_heads = self.opts.model['n_heads'],
            key_size = self.opts.model['key_size'],
            inner_dim = self.opts.model['inner_dim'],
            n_layers = self.opts.model['n_layers'],
            max_len = self.opts.model['max_len'],
            dropout = self.opts.model['dropout'],
            tied_emb_proj = self.opts.model['tied_emb_proj'],
            eps = self.opts.model['label_smoothing'],
            ctx_name = self.sl,
            layer_types = self.opts.model['dec_layer_types'],
            feat_name = self.feat_name,
            feat_dim = self.opts.model['feat_dim'],
            feat_ratio = self.opts.model['feat_ratio'],
            dropnet = self.opts.model['dropnet'],
        )

        if self.opts.model['shared_embs']:
            self.enc.embs[0].weight = self.dec.embs[0].weight
    
    def encode(self, batch, **kwargs):
        ctx_dict = super().encode(batch, **kwargs)

        if self.feat_name in batch:
            ctx_dict[self.feat_name] = (batch[self.feat_name], get_feat_mask(batch[self.feat_name]))

        return ctx_dict
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result
