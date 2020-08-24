#!/usr/bin/env python

import sys, os, subprocess
import argparse
import re
import torch
import glob
from pprint import pprint

def info(s, **kwargs):
    print(s, file=sys.stderr, **kwargs)

def tail(filename, n, default=''):
    return subprocess.check_output(['tail', f'-{n}', filename]).decode("utf-8")

def get(arr, idx, default=None):
    try:
        return arr[idx]
    except:
        return default

def flatten(raw_dict):
    flatten_dict = {}
    for sec_key, sec_value in raw_dict.items():
        if isinstance(sec_value, dict):
            for key, value in sec_value.items():
                flatten_dict['{}.{}'.format(sec_key, key)] = value
        else:
            flatten_dict[sec_key] = sec_value

    return flatten_dict

translate_regexp = re.compile(r'(?P<split>[^.]+).(no_unk.)?beam(?P<beam_size>\d+)$')
bleu_regexp = re.compile(r'(?P<split>[^.]+).(no_unk.)?beam(?P<beam_size>\d+)\.bleu$')
meteor_regexp = re.compile(r'(?P<split>[^.]+).(no_unk.)?beam(?P<beam_size>\d+)\.meteor$')

bleu_score_regex = re.compile(r'BLEU\s=\s(\d+\.\d+)')
meteor_score_regex = re.compile(r'METEOR\s=\s(\d+\.\d+)')

def get_opts(ckpt, load_ckeckpoint=False):
    flatten_opts = {
        'ckpt': ckpt,
        'train.exp_id': os.path.basename(ckpt).split('.')[0]
    }

    flatten_opts.update({f'subfolder.{i}': n for i, n in enumerate(os.path.dirname(ckpt).split('/'))})
    
    if load_ckeckpoint:
        ckpt_dict = torch.load(ckpt)
        opts = ckpt_dict['opts'].copy()
        del ckpt_dict
        flatten_opts.update(flatten(opts))

    return flatten_opts

def load_bleu_score(file):
    line = open(file).readlines()[0]
    score = get(bleu_score_regex.search(line), 1)
    return score

def load_meteor_score(file):
    line = open(file).readlines()[0]
    score = get(meteor_score_regex.search(line), 1)
    return score

def parse_results(ckpt):
    stat = {}
    for fname in glob.glob("{}.*".format(ckpt)):
        if translate_regexp.search(fname):
            m = translate_regexp.search(fname)
            split, beam_size = m['split'], m['beam_size']
            stat["{}.exists".format(split)] = fname
        
        elif bleu_regexp.search(fname):
            m = bleu_regexp.search(fname)
            split, beam_size = m['split'], m['beam_size']

            stat["{}.bleu".format(split)] = load_bleu_score(fname)
        
        elif meteor_regexp.search(fname):
            m = meteor_regexp.search(fname)
            split, beam_size = m['split'], m['beam_size']

            stat["{}.meteor".format(split)] = load_meteor_score(fname)

    return stat
    
def print_stat(stat, fields):
    fields = [stat.get(c, '') for c in fields]

    print(','.join(map(str, fields)))

def main(args):
    if (not args.no_header) and len(args.fields) > 0:
        print(','.join(args.fields))

    for ckpt in [c.strip() for c in args.checkpoints]:
        stat = get_opts(ckpt, args.load_checkpoint)
        stat.update(parse_results(ckpt))
        if len(args.fields) == 0:
            print('==> {}'.format(ckpt))
            pprint(stat)
        else:
            print_stat(stat, args.fields)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoints', nargs='*', default=sys.stdin)
    parser.add_argument('-f', '--fields', nargs='*', default=['ckpt'])
    parser.add_argument('--load-checkpoint', action='store_true', required=False)
    parser.add_argument('--no-header', action='store_true', required=False)

    args = parser.parse_args()
    info(args)

    main(args)