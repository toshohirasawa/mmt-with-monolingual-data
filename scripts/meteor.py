#!/usr/bin/env python
#Usage:
#  ls | meteor.py -l de /path/to/reference.de

import sys, os, argparse
from tempfile import NamedTemporaryFile
from nmtpytorch.metrics.meteor import METEORScorer

def info(*values):
    print(*values, file=sys.stderr)

METEORSCORER = METEORScorer()

def main(args):
    for pred in [p.strip() for p in args.predicts]:
        score_fname = '{}.meteor'.format(pred)
        result = METEORSCORER.compute(args.ref, pred, language=args.language)
        with open(score_fname, 'w') as fp:
            print(result, file=fp)
        info('{}: {}'.format(pred, result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('-l', '--language', default='auto')
    parser.add_argument('-p', '--predicts', default=sys.stdin)

    args = parser.parse_args()
    info(args)

    main(args)
