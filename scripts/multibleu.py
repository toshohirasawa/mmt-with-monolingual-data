#!/usr/bin/env python
import sys, os, argparse
from tempfile import NamedTemporaryFile
from nmtpytorch.metrics.multibleu import BLEUScorer

def info(*values):
    print(*values, file=sys.stderr)

BLEUSCORER = BLEUScorer()

# Corpus-level evaluation
def main(args):
    for pred in [p.strip() for p in args.predicts]:
        score_fname = '{}.bleu'.format(pred)
        result = BLEUSCORER.compute(args.refs, pred, language=args.language)
        with open(score_fname, 'w') as fp:
            print(result, file=fp)
        info('{}: {}'.format(pred, result))

# Sentence-level evaluation
def main_sentence(args):
    for pred in [p.strip() for p in args.predicts]:
        score_fname = '{}.bleu'.format(pred)
        result = eval_sent_bleu(args.refs, pred, language=args.language)
        with open(score_fname, 'w') as fp:
            for score in result:
                print(score, file=fp)

def eval_sent_bleu(hyps, refs, language):
    for hyp, ref in zip(open(hyps), open(refs)):
        with NamedTemporaryFile('w') as single_hyp_file, NamedTemporaryFile('w') as single_ref_file:
            single_hyp, single_ref = single_hyp_file.name, single_ref_file.name
            single_hyp_file.file.write(hyp)
            single_ref_file.file.write(ref)
            single_hyp_file.file.flush()
            single_ref_file.file.flush()
            score = BLEUSCORER.compute(refs=single_ref, hyps=single_hyp, language=language)
            yield score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('refs')
    parser.add_argument('-l', '--language', default=None)
    parser.add_argument('-p', '--predicts', default=sys.stdin)
    parser.add_argument('-s', '--sentence-level', action='store_true')

    args = parser.parse_args()
    info(args)

    assert os.path.exists(args.refs), \
        'Refernece file does not exists: {}'.format(args.refs)

    if not args.sentence_level:
        main(args)
    else:
        main_sentence(args)
