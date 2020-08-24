#!/usr/bin/env python
#Usage:
#  translate.py --eval-bleu --eval-meteor --ref ../../data/tok/test_2016_flickr.lc.norm.tok.de --lang de -k 5 -b 1000 -d 7

import sys, os, argparse

def info(*values):
    print(*values, file=sys.stderr)

def tail(name):
    with open(name, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    return last_line

def get_output_name(prefix, args):
    oname = '{}.{}'.format(prefix, args.split)
    if args.suppress_unk:
        oname += '.no_unk'
    oname += '.beam{}'.format(args.beam)
    return oname

def has_train_finished(ckpt):
    exp_id = os.path.basename(ckpt).split('.')[0]
    log_name = '{}/{}.log'.format(os.path.dirname(ckpt), exp_id)
    if not os.path.exists(log_name):
        # assume ckpt is copied from somewhere else
        return True
    else:
        last_line = tail(log_name)
        return last_line.startswith('Training finished on')

DEFAULT_OUTPUT_DIR='./output'
def main(args):
    checkpoints = [c.strip() for c in args.checkpoints]
    n_checkpoints = len(checkpoints)

    for i, ckpt in enumerate(checkpoints):
        dirname = os.path.dirname(ckpt)
        basename = os.path.basename(ckpt)

        output_prefix = args.output if args.output else ckpt
        oname = get_output_name(output_prefix, args)

        cmd = [
            'CUDA_VISIBLE_DEVICES={}'.format(args.device),
            'nmtpy', 'translate',
            '-b {}'.format(args.batch_size),
            '-k {}'.format(args.beam),
            '-m {}'.format(args.max_len),
            # '-a {}'.format(args.lp_alpha),
            '-s {}'.format(args.split),
            '-o {}'.format(output_prefix),
        ]
        if len(args.override)>0:
            cmd.append('-x {}'.format(' '.join(args.override)))
        if args.suppress_unk:
            cmd.append('-u')

        cmd += [ckpt, '1>&2']
        cmd = ' '.join(cmd)

        info(f'[{i+1}/{n_checkpoints}] ==> {cmd}')

        if args.dry_run:
            continue

        if not has_train_finished(ckpt):
            info('Training is in progress.')
            continue

        if os.path.exists(oname):
            info('Output file already exists.')
        else:
            os.system(cmd)

        print(oname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--checkpoints', default=sys.stdin)
    parser.add_argument('-b', '--batch-size', default=128)
    parser.add_argument('-k', '--beam', default=12)
    parser.add_argument('-m', '--max-len', default=100)
    # parser.add_argument('-a', '--lp-alpha', default=1)
    parser.add_argument('-d', '--device', default=0, required=False)
    parser.add_argument('-x', '--override', nargs="*", default=[])

    parser.add_argument('-s', '--split', default='test_2016_flickr')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-u', '--suppress-unk', action='store_true')
    
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()
    info(args)

    main(args)