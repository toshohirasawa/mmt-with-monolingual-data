import logging
logger = logging.getLogger('nmtpytorch')

import torch
from .am_transformer import AttentiveMultimodalTransformer
from ..layers import (
    SequentialImaginationDecoder, 
    SequentialImaginationDecoderV2
)

def get_aux_task_decoder(aux_task):
    return {
        'imagination':   SequentialImaginationDecoder,
        'imaginationv2': SequentialImaginationDecoderV2,
    }[aux_task.lower()]

class GradientNormalization(torch.nn.Module):
    def __init__(self, n_task=2, alpha=1.5):
        super().__init__()

        self.n_tasks = n_task
        self.alpha = alpha

        self.loss_weights = torch.nn.Parameter(torch.ones(self.n_tasks).float())
        self.initial_task_loss = None
    
    def forward(self, loss, aux_loss, n_items, W):
        # update standard forward loss
        # use constant version of loss_weights to avoid packprop from MT/Imag loss to loss_weight
        current_weights = self.loss_weights.tolist()
        loss = current_weights[0] * loss
        aux_loss = current_weights[1] * aux_loss

        # store L(0)
        if self.initial_task_loss is None:
            self.initial_task_loss = torch.stack([(loss / n_items).data, aux_loss.data])

        # weight task loss
        task_loss = torch.stack([loss / n_items, aux_loss])

        # compute Gw(t)
        Gw_t = []
        for i in range(self.n_tasks):
            gw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            Gw_t.append(torch.norm(self.loss_weights[i] * gw[0]))
        Gw_t = torch.stack(Gw_t)

        # compute r_i(t)
        task_loss_ratio = task_loss / self.initial_task_loss
        r_t = task_loss_ratio / torch.mean(task_loss_ratio)

        # compute /bar{G}w(t)
        mean_Gw_t = torch.mean(Gw_t)

        # GradNorm loss
        const_term = (mean_Gw_t * (r_t ** self.alpha)).data
        grad_norm_loss = torch.sum(torch.abs(Gw_t - const_term))

        return loss, aux_loss, grad_norm_loss

class MultitaskAttentiveMultimodalTransformer(AttentiveMultimodalTransformer):
    supports_beam_search = True

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'aux_task': 'imagination',
            'aux_loss_weight': 1.0,
            'no_aux_loss': False,
            'no_dec2img_loss': False,
            'use_feat_preds': False,
            # Imagination
            'loss_margin': 0.1,
            'img_dec_layers': 2,
            # gradient normalization
            'use_grad_norm': False,
            'grad_norm_alpha': 1.5,
            # Ablation
            'no_self_attn': False,
        })

    def __init__(self, opts):
        super().__init__(opts)

        self.aux_task = self.opts.model['aux_task']
        self.aux_loss_weight = self.opts.model['aux_loss_weight']

        self.no_aux_loss = self.opts.model['no_aux_loss']
        self.no_dec2img_loss = self.opts.model['no_dec2img_loss']

        self.use_feat_preds = self.opts.model['use_feat_preds']

        # GradNorm: only apply on training
        self.grad_norm = GradientNormalization(
            n_task=2,
            alpha=self.opts.model['grad_norm_alpha']
        ) if self.opts.model['use_grad_norm'] else None

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

        # predict features in encode, so can be used in the inference.
        if self.use_feat_preds or (self.training and (not self.no_aux_loss)):
            feat, feat_mask = ctx_dict.get(self.feat_name, [None, None])
            aux_result = self.img_dec(ctx_dict[self.sl], feat)

            if self.training and (not self.no_aux_loss):
                self.aux_loss[self.aux_task] = self.aux_loss_weight * aux_result['loss']
            
            if self.use_feat_preds:
                if feat_mask:
                    zero_feat_mask = (feat == 0.) * (feat_mask == 0).unsqueeze(-1)
                    feat[zero_feat_mask] = aux_result['preds'][zero_feat_mask]
                    # use 2 to distinguish preds from actual data
                    feat_mask[feat_mask == 0] = 2
                else:
                    feat = aux_result['preds']
                    feat_mask = torch.zeros(feat.shape[:2]).long().fill_(2).to(feat.device)

                # no backward from dec to img_dec through feature predictions.
                if self.no_dec2img_loss:
                    feat = feat.data
                
                ctx_dict[self.feat_name] = [feat, feat_mask]

        return ctx_dict

    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        # GradNorm: work but drop performance
        if self.training and self.grad_norm and (self.aux_task in self.aux_loss):
            result['loss'], self.aux_loss[self.aux_task], self.aux_loss['grad_norm'] = \
                self.grad_norm(result['loss'], self.aux_loss[self.aux_task], result['n_items'], self.enc.get_last_layer())

        return result
