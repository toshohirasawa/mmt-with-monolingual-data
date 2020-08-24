#!/usr/bin/env python
# original all-but-the-top code:
# https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390

import sys, os
import logging
import argparse
logger = logging.getLogger(__name__)

import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import torch

def all_but_the_top(v, D):
    """
    Arguments:
        :v: word vectors of shape (n_words, n_dimensions)
        :D: number of principal components to subtract
    """
    # 1. Subtract mean vector
    v_tilde = v - np.mean(v, axis=0)

    # 2. Compute the first `D` principal components
    #    on centered embedding vectors
    pca = PCA(n_components=D)
    pca = pca.fit(v_tilde)

    # Subtract first `D` principal components
    # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
    emb_pca = pca.transform(v_tilde).reshape(-1, D, 1) * \
        pca.components_.reshape(1, D, -1)
    emb_pca = emb_pca.sum(axis=-2)
    v_hat = v_tilde - emb_pca

    return v_hat

# main
def apply_all_but_the_top(input_file: str, output_file: str, n_comp: int):
    vectors = np.load(input_file)

    slice_id = 3 if args.exclude_eos else 2
    special_embs = vectors[:slice_id]
    word_embs = vectors[slice_id:]

    word_embs = all_but_the_top(word_embs, n_comp)
    v_hat = np.vstack((special_embs, word_embs))

    # norm
    # use torch to avoid "divide by zero" error
    if not args.skip_norm:
        v_hat = torch.from_numpy(v_hat)
        v_hat = torch.functional.F.normalize(v_hat)
        v_hat = v_hat.numpy()

    np.save(output_file, v_hat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, 
        help="Path to input file in npy format")
    parser.add_argument("-o", "--output", required=True, 
        help="Path to output file")
    parser.add_argument("-d", "--n-components", required=False, 
        type=int, default=3, help="Num of PCA components to substruct.")
    parser.add_argument("--exclude-eos", required=False,
        action="store_true", help="")
    parser.add_argument("--skip-norm", required=False,
        action="store_true", help="")

    args = parser.parse_args()
    print(args, file=sys.stderr)

    apply_all_but_the_top(args.input, args.output, args.n_components)
