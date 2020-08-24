#!/usr/bin/env python

import sys, os
import argparse
import numpy as np

# main
def main(args):
    x = np.load(args.input)
    print(f'Data shape is {x.shape}')
    
    # norm (replace 0. by 1. to avoid dividing by zero)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm[x_norm==0] = 1.
    normed_x = x / x_norm

    np.save(args.output, normed_x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    print(args, file=sys.stderr)

    main(args)
