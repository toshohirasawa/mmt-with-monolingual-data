import logging

import torch
from torch import nn
import numpy as np
import math

from .transformer import Transformer
from ..layers import SequentialImaginationDecoder
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

class SequentialImaginationTransformer(Transformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'feat_name': 'feats',       # name
            'feat_dim': 1024,           # dimension
            'n_feats': 10,
            'loss_margin': 0.1,         #
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
        self.img_dec = SequentialImaginationDecoder(
            ctx_dim=self.opts.model['model_dim'],
            feat_dim=self.opts.model['feat_dim'],
            n_feats=self.opts.model['n_feats'],
            margin=self.opts.model['loss_margin'],
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
            aux_result = self.img_dec(ctx_dict[self.sl], ctx_dict[self.feat_name][0])
            self.aux_loss['imagination'] = aux_result['loss']

        return result
