import math
import numpy as np
import fasttext
from gensim.models import KeyedVectors
from tqdm import tqdm

import logging
logger = logging.getLogger('nmtpytorch')

def uniform(dim, bias=None):
    stdv = 1. / math.sqrt(dim)
    x = np.random.uniform(-stdv, stdv, dim)
    if bias is not None:
        x += bias
    return x

def load_fasttext(bin_fname, vocab):
    bin_fname = str(bin_fname)
    logger.info(f'loading FastText embeddings from {bin_fname}')
    
    model = fasttext.load_model(bin_fname)
    dim = model.get_dimension()

    # word embeddings
    word_embs = [model.get_word_vector(w) for w in list(vocab.tokens())[4:]]

    # UNK (the mean vector of OOV words w.r.t. model vocab)
    unk_emb = [model.get_word_vector(w) for w in tqdm(model.get_words()) if w not in vocab._map]
    assert len(unk_emb) > 0, 'Cannot construct the embedding for unknown words.'
    unk_emb = np.stack(unk_emb).mean(axis=0)

    # special tokens: BOS (uniformaly distributed around centroid)
    pad_emb = np.zeros(dim)
    eos_emb = model.get_word_vector('</s>')
    bos_emb = uniform(dim, np.append([unk_emb, eos_emb], word_embs, axis=0).mean(axis=0))
    
    return np.append([pad_emb, bos_emb, eos_emb, unk_emb], word_embs, axis=0)

def load_word2vec(bin_fname, vocab):
    bin_fname = str(bin_fname)
    logger.info(f'loading word2vec embeddings from {bin_fname}')

    binary = bin_fname.endswith('.bin')
    model = KeyedVectors.load_word2vec_format(bin_fname, binary=binary)
    dim = model.vector_size

    # PAD
    pad_emb = np.zeros(dim)

    # EOS
    eos_emb = model.get_vector('</s>')

    # UNK
    unk_emb = [model.get_vector(w) for w in tqdm(model.vocab.keys()) if w not in vocab._map]
    assert len(unk_emb) > 0, 'Cannot construct the embedding for unknown words.'
    unk_emb = np.stack(unk_emb).mean(axis=0)

    # Other types
    # assign UNK embedding for words not in word2vec model
    embs = []
    for w in list(vocab.tokens())[4:]:
        if w in model.vocab:
            embs.append(model.get_vector(w))
        else:
            logger.info(f'Unknown word: {w}')
            embs.append(unk_emb.copy())

    # BOS
    bos_emb = uniform(dim, np.append([eos_emb, unk_emb], embs, axis=0).mean(axis=0))

    return np.append([pad_emb, bos_emb, eos_emb, unk_emb], embs, axis=0)

def load_glove(bin_fname, vocab):
    '''
    Use word2vec-format version of GloVe embedding by applying gensim.scripts.glove2word2vec.
    '''
    bin_fname = str(bin_fname)
    logger.info(f'loading Glove embeddings from {bin_fname}')

    model = KeyedVectors.load_word2vec_format(bin_fname, binary=bin_fname.endswith('.bin'))
    dim = model.vector_size

    # PAD
    pad_emb = np.zeros(dim)

    # UNK
    unk_emb = [model.get_vector(w) for w in tqdm(model.vocab.keys()) if w not in vocab._map]
    assert len(unk_emb) > 0, 'Cannot construct the embedding for unknown words.'
    unk_emb = np.stack(unk_emb).mean(axis=0)

    # Other types
    # assign UNK embedding for words not in word2vec model
    embs = []
    for w in list(vocab.tokens())[4:]:
        if w in model.vocab:
            embs.append(model.get_vector(w))
        else:
            logger.info(f'Unknown word: {w}')
            embs.append(unk_emb.copy())

    # BOS
    bos_emb = uniform(dim, np.append([unk_emb], embs, axis=0).mean(axis=0))

    # EOS
    eos_emb = uniform(dim, np.append([unk_emb], embs, axis=0).mean(axis=0))

    return np.append([pad_emb, bos_emb, eos_emb, unk_emb], embs, axis=0)