#!/usr/bin/env python

import sys, os
import logging
import argparse

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from fasttext import FastText

logger = logging.getLogger(__name__)

def fasttext2word2vec(fasttext_input_file: str, word2vec_output_file: str):
    fasttext_model = FastText.load_model(fasttext_input_file)

    n_lines, n_dims = len(fasttext_model.labels), fasttext_model.get_dimension()
    logger.info("converting %i vectors from %s to %s", n_lines, fasttext_input_file, word2vec_output_file)

    glovekv = KeyedVectors(vector_size=n_dims)
    entries = fasttext_model.labels
    weights = [fasttext_model.get_word_vector(w) for w in entries]
    glovekv.add(entries, weights)

    glovekv.save_word2vec_format(word2vec_output_file, binary=word2vec_output_file.endswith('.bin'))

    return n_lines, n_dims

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input file in FastText format")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    logger.info(args)

    fasttext2word2vec(args.input, args.output)
