# -*- coding: utf-8 -*-
import pickle as pkl

import torch
from torch import nn
from transformers import pipeline
from nmtpytorch.vocabulary import Vocabulary

def get_partial_embedding_layer(vocab, embedding_dim, pretrained_file,
                                freeze='none', oov_zero=True):
    """A partially updateable embedding layer with pretrained embeddings.
    This is experimental and not quite tested."""
    avail_idxs, miss_idxs = [], []
    avail_embs = []

    # Load the pickled dictionary
    with open(pretrained_file, 'rb') as f:
        pret_dict = pkl.load(f)

    for idx, word in vocab._imap.items():
        if word in pret_dict:
            avail_embs.append(pret_dict[word])
            avail_idxs.append(idx)
        else:
            miss_idxs.append(idx)

    # This matrix contains the pretrained embeddings
    avail_embs = torch.Tensor(avail_embs)

    # We don't need the whole dictionary anymore
    del pret_dict

    n_pretrained = len(avail_idxs)
    n_learned = vocab.n_tokens - n_pretrained

    # Sanity checks
    assert len(avail_idxs) + len(miss_idxs) == vocab.n_tokens

    # Create the layer
    emb = nn.Embedding(vocab.n_tokens, embedding_dim, padding_idx=0)
    if oov_zero:
        emb.weight.data.fill_(0)

    # Copy in the pretrained embeddings
    emb.weight.data[n_learned:] = avail_embs
    # Sanity check
    assert torch.equal(emb.weight.data[-1], avail_embs[-1])

    grad_mask = None
    if freeze == 'all':
        emb.weight.requires_grad = False
    elif freeze == 'partial':
        # Create bitmap gradient mask
        grad_mask = torch.ones(vocab.n_tokens)
        grad_mask[n_learned:].fill_(0)
        grad_mask[0].fill_(0)
        grad_mask.unsqueeze_(1)

        def grad_mask_hook(grad):
            return grad_mask.to(grad.device) * grad

        emb.weight.register_hook(grad_mask_hook)

    # Return the layer
    return emb

def init_nn_embedding(**kwargs):
    keywords = ['num_embeddings', 'embedding_dim', 'padding_idx',
                'max_norm', 'norm_type', 'scale_grad_by_freq',
                'sparse', '_weight']
    effective_kwargs = {k:v for k, v in kwargs.items() if k in keywords}
    return nn.Embedding(**effective_kwargs)

class BertEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab: Vocabulary, model: str, model_dim: int, **kwargs):
        super().__init__()

        self.vocab = vocab
        self.model = model
        self.pipeline = pipeline('feature-extraction', model=self.model, device=torch.cuda.current_device())
        self.transform = nn.Linear(in_features=model_dim, out_features=embedding_dim)
    
    def forward(self, input: torch.Tensor):
        sents = self.vocab.list_of_idxs_to_sents(input.t().tolist())

        raw_embs = self.pipeline(sents)
        raw_embs = torch.Tensor(raw_embs).to(input.device).transpose(0,1)
        raw_embs = torch.autograd.Variable(raw_embs)

        embs = self.transform(raw_embs)
        return embs

def get_embedding(type_):
    return {
        'nn': init_nn_embedding,
        'bert': BertEmbedding
    }[type_]

class EmbeddingOutput(nn.Module):

    def __init__(self, n_vocab, emb_size, weight=None):
        super().__init__()
        self.n_vocab, self.emb_size = n_vocab, emb_size

        if weight == None:
            weight = torch.randn(n_vocab, emb_size)
        else:
            # travarse matrix for easy matmul calclulation in forward
            self.weight = nn.Parameter(weight)
            self.weight.requires_grad = False
        
    def forward(self, input):
        # normalize
        input_norm = input.norm(dim=-1, keepdim=True)

        emb_norm = self.weight.data.norm(dim=-1, keepdim=True)
        emb_norm[emb_norm == 0.] = 1.

        input = input / input_norm
        emb = self.weight.data / emb_norm

        # grad in this module makes learning unstable
        return input.matmul(emb.t())

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(n_vocab={self.n_vocab}, emb_size={self.emb_size})'