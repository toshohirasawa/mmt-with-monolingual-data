# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from .imagination import Imagination
from .nmt import NMT
from ..layers import TextEncoder
from ..layers.decoders import get_decoder, SequentialImaginationDecoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class SequentialImagination(NMT):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'feat_name': 'feats',       # name
            'feat_dim': 2048,           # dimension
            'n_feats': 10,
            'loss_margin': 0.1,         #
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.feat_name = self.opts.model['feat_name']
        self.ctx_sizes[self.feat_name] = self.opts.model['feat_dim']

    def setup(self, is_train=True):
        super().setup(is_train)
        
        # Imagination
        self.img_dec = SequentialImaginationDecoder(
            ctx_dim=self.ctx_sizes[self.sl],
            feat_dim=self.ctx_sizes[self.feat_name],
            n_feats=self.opts.model['n_feats'],
            margin=self.opts.model['loss_margin'],
        )

    def encode(self, batch, **kwargs):
        d = super().encode(batch, **kwargs)
        d[self.feat_name] = (batch[self.feat_name], None)
        return d

    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)

        # Get loss dict
        result = self.dec(ctx_dict, batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        if self.training:
            imag_loss = self.img_dec(ctx_dict[self.sl], ctx_dict[self.feat_name][0])
            self.aux_loss['imagination'] = imag_loss['loss']

        return result
