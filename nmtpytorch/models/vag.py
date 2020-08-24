# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from ..layers import TextEncoder
from ..layers.decoders import get_decoder
from ..layers import VAGConditionalDecoder, VAGSharedSpaceDecoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.nn import get_activation_fn
from ..datasets import MultimodalDataset
from ..metrics import Metric
from ..utils.scheduler import Scheduler

logger = logging.getLogger('nmtpytorch')

class VisualTextAttention(nn.Module):
    """Attention layer with dot product."""
    def __init__(self, hid_dim, ctx_dim):
        # NOTE:
        # mlp_bias here to not break models that pass mlp_bias to all types
        # of attentions
        super().__init__()

        self.ctx_dim = ctx_dim
        self.hid_dim = hid_dim

        # Adaptor from RNN's hidden dim to mid_dim
        self.hid2ctx = nn.Sequential(
            nn.Linear(self.hid_dim, self.ctx_dim, bias=False),
            nn.Tanh()
        )

        # Additional context projection within same dimensionality
        self.ctx2ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, self.ctx_dim, bias=False),
            nn.Tanh()
        )

    def forward(self, hid, ctx, ctx_mask=None):
        # SxBxC
        ctx_ = self.ctx2ctx(ctx)
        # TxBxC
        hid_ = self.hid2ctx(hid)

        # shuffle dims to prepare for batch mat-mult -> SxB
        scores = torch.bmm(hid_.permute(1, 0, 2), ctx_.permute(1, 2, 0)).squeeze(1).t()

        # Normalize attention scores correctly -> S*B
        if ctx_mask is not None:
            # Mask out padded positions with -inf so that they get 0 attention
            scores.masked_fill_((1 - ctx_mask).bool(), -1e8)

        alpha = F.softmax(scores, dim=0)

        # Transform final context vector to H for further decoders
        return alpha, (alpha.unsqueeze(-1) * ctx).sum(0)

class VisualAttentionGrounding(nn.Module):
    '''
    Implementation of Zhou et al., 2018
    "A Visual Attention Grounding Neural Model for Multimodal Machine Translation"
    '''
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            # Text related options
            'emb_type': 'nn',
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'emb_model': None,
            'emb_model_dim': None,
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'n_decoders': 1,
            'img_preprop_type': 'none',
            'loss_type': 'maxmargin',   # Loss type for decoder
            'loss_args': {},
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       #
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'bos_type': 'emb',          # 'emb': default learned emb
            'bos_activ': None,          #
            'bos_dim': None,            #

            # # Image related options
            'feat_name': 'feats',   # name
            'feat_dim': 2048,       # depends on the features used
            'mtl_alpha': 0.99,      # weight for mt task loss 
            'dec_init_lambda': 0.5,
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Vocabulary
        self.vocabs = {}
        for lang, fname in self.opts.vocabulary.items():
            self.vocabs[lang] = Vocabulary(fname, name=lang)
        
        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])
        self.sl = self.topology.get_src_langs()[0]
        self.src_vocab = self.vocabs[self.sl]
        self.n_src_vocab = len(self.src_vocab)
        self.tl = self.topology.get_trg_langs()[0]
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)

        # references
        # TODO: valuation data sould be excluded from model
        self.val_refs = self.opts.data['val_set'][self.tl]


        # MT model parameters

        # Textual context size
        # it should be (enc_dim * 2) as it is the concat of forward and backward
        if 'enc_dim' in self.opts.model:
            self.ctx_sizes = {str(self.sl): self.opts.model['enc_dim'] * 2}


        # Check tying option
        if self.opts.model['tied_emb'] not in [False, '2way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        # Visual-related parameters
        self.feat_name = self.opts.model['feat_name']
        self.ctx_sizes[self.feat_name] = self.opts.model['feat_dim']
        self.mtl_alpha = self.opts.model['mtl_alpha']

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        if hasattr(self.enc.emb, 'weight'):
            with torch.no_grad():
                self.enc.emb.weight.data[0].fill_(0)
        
    def setup(self, is_train=True):
        # Shared encoder
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_type=self.opts.model['emb_type'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            emb_model=self.opts.model['emb_model'],
            emb_model_dim=self.opts.model['emb_model_dim'],
            layer_norm=self.opts.model['enc_lnorm'],
            vocab=self.src_vocab,)

        # Vision-Text Attention
        self.vis_att = VisualTextAttention(
            self.ctx_sizes[self.feat_name], 
            self.ctx_sizes[self.sl]
        )

        # MT decoder
        self.dec = VAGConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            dec_init_lambda=self.opts.model['dec_init_lambda'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # IMAGINATION decoder
        self.img_dec = VAGSharedSpaceDecoder(
            txt_ctx_size=self.ctx_sizes[self.sl],    # bidirectional
            img_ctx_size=self.ctx_sizes[self.feat_name],
            output_size=512
        )

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            preprocess_type=self.opts.model['img_preprop_type'])
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        ctx_dict = {
            str(self.sl): self.enc(batch[self.sl]),
            self.feat_name: (batch[self.feat_name], None)
        }

        # compute vector for sentence
        vis_att, txt_ctx = self.vis_att(ctx_dict[self.feat_name][0], *ctx_dict[self.sl])
        ctx_dict['sent'] = (txt_ctx.unsqueeze(0), None)

        return ctx_dict

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
            # Text representation using Vision-Text Attention
            img_loss = self.img_dec(ctx_dict['sent'], ctx_dict[self.feat_name])
            self.aux_loss['vag'] = (1. - self.mtl_alpha) * img_loss['loss']

            result['loss'] = self.mtl_alpha * result['loss']

        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch, is_train=False)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def get_decoder(self, task_id=None):
        return self.dec

    def get_enc_emb(self):
        return self.enc.emb
    
    def get_dec_emb(self):
        return self.dec.emb