# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from .nmt import NMT
from ..layers.decoders import ImaginationDecoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class Imagination(NMT):
    def set_defaults(self):
        super().set_defaults()

        self.defaults.update({
            'imag_activ': 'tanh',
            'imag_loss_margin': 0.1,
            'imag_z_w': 1,
        })
    
    def __init__(self, opts):
        super().__init__(opts)

        self.z_w = self.opts.model['imag_z_w']

    def setup(self, is_train=True):
        super().setup(is_train)

        self.img_dec = ImaginationDecoder(
            ctx_dim=self.ctx_sizes[self.sl],
            feat_dim=self.opts.model['feat_dim'],
            att_activ=self.opts.model['imag_activ'],
            margin=self.opts.model['imag_loss_margin'],
        )

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        ctx_dict = self.encode(batch)

        result = self.dec(ctx_dict, batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        if self.training:
            imag_loss = self.img_dec(ctx_dict[self.sl], ctx_dict[self.feat_name])
            self.aux_loss['imagination'] = self.z_w * imag_loss['loss']

        return result