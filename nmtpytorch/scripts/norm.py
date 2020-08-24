#!/usr/bin/env python

import sys, os, logging, argparse
logger = logging.getLogger(__name__)

import numpy as np

# main
def main(args):
    raw_data = np.load(args.input)
    pad_vec = raw_data[0]
    tok_vec = raw_data[1:, :]
    
    bias = tok_vec.mean(axis=0)
    c_tok_vec = tok_vec - bias

    l2_norm = np.linalg.norm(c_tok_vec, axis=args.dim, keepdims=True)
    l2_norm[l2_norm == 0] = 1

    print(f'Before: bias={bias.mean()}, l2_norm={l2_norm.mean()}')
    
    n_tok_vec = c_tok_vec / l2_norm

    normed_bias = n_tok_vec.mean(axis=0).mean()
    normed_l2_norm = np.linalg.norm(n_tok_vec, axis=args.dim).mean()
    print(f'After: bias={normed_bias}, l2_norm={normed_l2_norm}')

    n_data = np.vstack([pad_vec, n_tok_vec])

    np.save(args.output, n_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
        help="Path to input file")
    parser.add_argument('-o', '--output', required=True,
        help="Path to output file")
    parser.add_argument('-d', '--dim', default=-1, required=False,
        help="")

    args = parser.parse_args()
    print(args)

    main(args)
