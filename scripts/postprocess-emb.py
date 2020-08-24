#!/usr/bin/env python

import sys, os
import argparse

import numpy as np

from nmtpytorch.utils.embedding_postprocess import apply_abtt, apply_lc
from nmtpytorch.vocabulary import Vocabulary

# main
def main(args):
    if args.type == 'abtt':
        embs = apply_abtt(np.load(args.input), applys_norm=args.applys_norm)
    elif args.type == 'lc':
        embs = apply_lc(np.load(args.input), applys_norm=args.applys_norm)
    elif args.type == 'autoencoder':
        pass
    else:
        raise Exception()

    np.save(args.output, embs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-t', '--type', choices=['abtt', 'lc', 'autoencoder'], required=True)
    parser.add_argument('--applys-norm', action='store_true', required=False)

    args = parser.parse_args()
    print(args, file=sys.stderr)

    main(args)
