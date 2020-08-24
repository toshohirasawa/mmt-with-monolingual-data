#!/usr/bin/env python

import sys, os, argparse
from datetime import datetime
from tqdm import tqdm

def info(*values):
    print(*values, file=sys.stderr)

def tail(name):
    with open(name, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    return last_line

def has_train_finished(log_name):
    if not os.path.exists(log_name):
        # assume ckpt is copied from somewhere else
        return True
    else:
        last_line = tail(log_name)
        return last_line.startswith('Training finished on')

def main(args):
    for log_name in tqdm([l.strip() for l in args.logs]):
        if not has_train_finished(log_name):
            mtime = datetime.fromtimestamp(os.path.getmtime(log_name))
            print('==> {} (Last Updated: {})'.format(log_name, mtime))
            os.system('tail -n5 {}'.format(log_name))
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', default=sys.stdin)

    args = parser.parse_args()
    info(args)

    main(args)