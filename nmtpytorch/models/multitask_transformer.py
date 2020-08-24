import logging
logger = logging.getLogger('nmtpytorch')

import torch
from ..utils.misc import get_n_params, get_feat_mask
from .transformer import Transformer
from ..layers import (
    SequentialImaginationDecoder,
)

def get_aux_task_decoder(aux_task):
    return {
        'imagination':   SequentialImaginationDecoder,
    }[aux_task.lower()]

class MultitaskTransformer(Transformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'aux_task': 'imagination',
            'aux_loss_weight': 1.0,
            # visual features
            'feat_name': 'feats',
            'feat_dim': 2048,
            'n_feats': 18,
            # Imagination
            'loss_margin': 0.1,
            'img_dec_layers': 2,
            # Ablation
            'no_self_attn': False,
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.aux_task = self.opts.model['aux_task']
        self.aux_loss_weight = self.opts.model['aux_loss_weight']

        self.feat_name = self.opts.model['feat_name']

    def reset_parameters(self):
        super().reset_parameters()

    def setup(self, is_train=True):
        super().setup(is_train) # self.enc, self.dec

        self.img_dec = get_aux_task_decoder(self.aux_task)(
            ctx_dim=self.opts.model['model_dim'],
            feat_dim=self.opts.model['feat_dim'],
            n_feats=self.opts.model['n_feats'],
            margin=self.opts.model['loss_margin'],
            n_layers=self.opts.model['img_dec_layers'],
            no_self_attn=self.opts.model['no_self_attn'],
        )
    
    def encode(self, batch, **kwargs):
        ctx_dict = super().encode(batch, **kwargs)

        if self.feat_name in batch:
            ctx_dict[self.feat_name] = (batch[self.feat_name], get_feat_mask(batch[self.feat_name]))
        
        # predict features in encode, so can be used in the inference.
        feat, feat_mask = ctx_dict.get(self.feat_name, [None, None])
        aux_result = self.img_dec(ctx_dict[self.sl], feat)

        ctx_dict[f'{self.feat_name}_pred'] = [aux_result['pred'], None]

        if self.training and 'loss' in aux_result:
            self.aux_loss[self.aux_task] = self.aux_loss_weight * aux_result['loss']

        return ctx_dict

    def forward(self, batch, **kwargs):
        # initialize aux_loss
        self.aux_loss = {}

        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result
