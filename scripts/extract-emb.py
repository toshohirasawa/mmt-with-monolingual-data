#!/usr/bin/env python

import sys, os
import argparse

import numpy as np

from nmtpytorch.utils.embedding import load_fasttext,load_glove,load_word2vec
from nmtpytorch.vocabulary import Vocabulary

# main
def main(args):
    vocab = Vocabulary(args.vocab, None)
    
    if args.type == 'fastText':
        embs = load_fasttext(args.embedding, vocab)
    elif args.type == 'glove':
        embs = load_glove(args.embedding, vocab)
    elif args.type == 'word2vec':
        embs = load_word2vec(args.embedding, vocab)
    else:
        raise Exception()

    np.save(args.output, embs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', type=str, required=True)
    parser.add_argument('-e', '--embedding', type=str, required=True)
    parser.add_argument('-t', '--type', choices=['fastText', 'glove', 'word2vec'], required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    print(args, file=sys.stderr)

    main(args)
