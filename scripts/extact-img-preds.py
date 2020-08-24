#!/usr/bin/env python

import sys, os, logging, argparse
from tempfile import NamedTemporaryFile

from nmtpytorch.config import Options
from nmtpytorch.utils.misc import load_pt_file, pbar
from nmtpytorch.utils.data import make_dataloader
from nmtpytorch.utils.device import DEVICE

from nmtpytorch import models

import numpy as np

def info(msg):
    print(msg, file=sys.stderr)

# main
def main(args):
    # load a model
    weights, _, opts = load_pt_file(args.checkpoint)
    opts = Options.from_dict(opts)
    instance = getattr(models, opts.train['model_type'])(opts=opts)

    instance.setup(is_train=False)
    instance.load_state_dict(weights, strict=False)
    instance.to(DEVICE)
    instance.train(False)

    # restrict to load only src text data
    src_ds   = instance.topology.get_src_langs()[0]
    src_lang = src_ds.data
    instance.topology.all  = {src_lang: src_ds}
    instance.topology.srcs = {src_lang: src_ds}
    instance.topology.trgs = {}

    # load data into dataset
    instance.opts.data['tmp_set'] = {
        src_lang: args.text_file
    }

    dataset = instance.load_data('tmp', opts.train['batch_size'], mode='beam')
    loader = make_dataloader(dataset)

    info('Starting translation')
    output_data = []
    for batch in pbar(loader, unit='batch'):
        batch.device(DEVICE)
        
        ctx_dict = instance.encode(batch, text_only=True)

        preds = instance.img_dec.predict(ctx_dict[src_lang])
        output_data.append(preds.data.numpy())
    
    info('Saving prediction to {}'.format(args.output))
    data = np.vstack(output_data)
    np.save(args.output, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('-t', '--text-file')
    parser.add_argument('checkpoint')

    args = parser.parse_args()
    info(args)

    main(args)
