import logging

import torch
from torch import nn
import numpy as np
import math

from .transformer import Transformer
from ..layers import ImaginationDecoder
from ..utils.misc import get_n_params
from ..utils.data import sort_batch
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')

class ImaginationTransformer(Transformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'feat_name': 'feats',       # name
            'feat_dim': 2048,           # dimension
            'imag_pool_type': 'max',    #
            'imag_activ': 'tanh',       #
            'imag_loss_margin': 0.1,    #
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.feat_name = self.opts.model['feat_name']

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()
        self.img_dec.reset_parameters()

    def setup(self, is_train=True):
        super().setup(is_train)
        # Imagination decoder here
        self.img_dec = ImaginationDecoder(
            ctx_dim=self.opts.model['model_dim'],
            feat_dim=self.opts.model['feat_dim'],
            pool_type=self.opts.model['imag_pool_type'],
            att_activ=self.opts.model['imag_activ'],
            margin=self.opts.model['imag_loss_margin'],
        )

    def encode(self, batch, **kwargs):
        d = super().encode(batch, **kwargs)
        d[self.feat_name] = (batch[self.feat_name], None)
        return d
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        if self.training:
            aux_result = self.img_dec(ctx_dict[self.sl], ctx_dict[self.feat_name])
            self.aux_loss['imagination'] = aux_result['loss']

        return result
