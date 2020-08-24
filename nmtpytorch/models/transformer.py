import logging

import torch
from torch import nn
import numpy as np
import math

from ..layers import TransformerEncoder, TransformerDecoder
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

class Transformer(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults ={
            'direction': None,
            'model_dim': 512,
            'n_heads': 8,
            'key_size': 64,
            'inner_dim': 2048,
            'n_layers': 6,
            'max_len': 100,
            'dropout': 0.3,
            'tied_emb_proj': True,           # tie embedding and projection weights
            'shared_embs': False,        # share embedding between encoder and decoder
            'label_smoothing': 0.1,
            'sampler_type': 'token',
            'bucket_by': None,
            'bucket_order': None,
        }

    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.opts.model = self.set_model_options(opts.model)
        self.topology = Topology(self.opts.model['direction'])
        self.vocabs = {n: Vocabulary(fn, name=n) for n, fn in self.opts.vocabulary.items()}
        
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            self.tl = tlangs[0]
            self.trg_vocab = self.vocabs[self.tl]
            self.n_trg_vocab = len(self.trg_vocab)
            self.val_refs = self.opts.data['val_set'][self.tl]
        
        self.aux_loss = {}

    def __repr__(self):
        s = [super().__repr__()]
        for vocab in self.vocabs.values():
            s.append(str(vocab))
        s.append(str(get_n_params(self)))

        return '\n'.join(s)

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
        self.enc.reset_parameters()
        self.dec.reset_parameters()

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
        self.dec = TransformerDecoder(
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
        )

        if self.opts.model['shared_embs']:
            self.enc.embs[0].weight = self.dec.embs[0].weight

    def load_data(self, split, batch_size, mode='train'):
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'],
            n_feats=self.opts['model'].get('n_feats', None))
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])
    
    def encode(self, batch, **kwargs):
        ctx_dict = {k: [v, None] for k, v in batch.items() if k not in (self.sl, self.tl)}
        
        hs, x, attn = self.enc(batch[self.sl], batch=batch)
        ctx_dict[self.sl] = [hs, x]

        return ctx_dict
    
    def forward(self, batch, **kwargs):
        ctx_dict = self.encode(batch)
        y = batch[self.tl]

        result = self.dec(ctx_dict, y)
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        return result

    def test_performance(self, data_loader, dump_file=None):
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]
    
    def get_decoder(self, task_id=None):
        return self.dec
