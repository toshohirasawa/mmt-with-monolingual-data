#!/usr/bin/env python

import sys, os, logging, argparse
logger = logging.getLogger(__name__)

import numpy as np
import math
from gensim.models import KeyedVectors
from nmtpytorch.vocabulary import Vocabulary
import fasttext
from tqdm import tqdm

def uniform(dim, bias=None):
    stdv = 1. / math.sqrt(dim)
    x = np.random.uniform(-stdv, stdv, dim)
    if bias is not None:
        x += bias
    return x

def load_word2vec(word2vec_input_file: str, vocab_file: str, bos=None, eos=None):
    kv = KeyedVectors.load_word2vec_format(word2vec_input_file)
    vocab = Vocabulary(vocab_file, '')

    emb_dim = kv.vector_size
    emb_bias = kv.vectors.mean(axis=0)
    
    unk_emb = [kv.get_vector(w) for w in tqdm(kv.vocab.keys()) if w not in vocab._map]
    if len(unk_emb) > 0:
        unk_emb = np.stack(unk_emb).mean(axis=0)
    else:
        unk_emb = uniform(emb_dim, emb_bias)
    
    bos_emb = kv.get_vector(bos) if bos else uniform(emb_dim, emb_bias)
    eos_emb = kv.get_vector(eos) if eos else uniform(emb_dim, emb_bias)
    pad_emb = np.zeros(emb_dim)

    words = list(vocab.tokens())[4:]
    word_embs = [
        kv.get_vector(w) if w in kv.vocab else unk_emb.copy() for w in words
    ]

    return np.append([pad_emb, bos_emb, eos_emb, unk_emb], word_embs, axis=0)

def load_fasttext(fasttext_input_file: str, vocab_file: str, bos=None, eos=None):
    ft = fasttext.load_model(fasttext_input_file)
    vocab = Vocabulary(vocab_file, '')

    emb_dim = ft.get_dimension()
    emb_bias = ft.get_output_matrix().mean(axis=0)

    unk_emb = [ft.get_word_vector(w) for w in tqdm(ft.labels) if w not in vocab._map]
    if len(unk_emb) > 0:
        unk_emb = np.stack(unk_emb).mean(axis=0)
    else:
        unk_emb = uniform(emb_dim, emb_bias)
    
    bos_emb = ft.get_word_vector(bos) if bos else uniform(emb_dim, emb_bias)
    eos_emb = ft.get_word_vector(eos) if eos else uniform(emb_dim, emb_bias)
    pad_emb = np.zeros(emb_dim)

    words = list(vocab.tokens())[4:]
    word_embs = [ft.get_word_vector(w) for w in words]

    return np.append([pad_emb, bos_emb, eos_emb, unk_emb], word_embs, axis=0)

# main
def main(args):
    embs = {
        'word2vec': load_word2vec,
        'fasttext': load_fasttext,
    }[args.type](args.input, args.vocab, bos=args.bos, eos=args.eos)

    np.save(args.output, embs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="Path to input file")
    parser.add_argument('-v', '--vocab', required=True, help="Path to vocab file")
    parser.add_argument('-o', '--output', required=True, help="Path to output file")
    parser.add_argument('-t', '--type', required=True, choices=['fasttext', 'word2vec'])
    parser.add_argument('--bos', required=False, default=None, type=str)
    parser.add_argument('--eos', required=False, default=None, type=str)

    args = parser.parse_args()
    logger.info(args)

    main(args)
