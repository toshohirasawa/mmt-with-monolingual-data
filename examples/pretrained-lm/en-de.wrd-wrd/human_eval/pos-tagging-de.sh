#!/bin/bash -eu

# Use https://github.com/rsennrich/ParZu to pos-tag mt outputs

mkdir -p parze
mkdir -p pos
mkdir -p stts-pos

for mt_type in 'text-only' 'bert' 'lxmert' 'all-inclusive'; do
    if [ ! -f ./parzu/test_2016_flickr.lc.norm.tok.${mt_type}.de ]; then
        cat tok/test_2016_flickr.lc.norm.tok.${mt_type}.de \
            | docker run -i rsennrich/parzu /ParZu/parzu \
            >./parzu/test_2016_flickr.lc.norm.tok.${mt_type}.de
    fi
    cat ./parzu/test_2016_flickr.lc.norm.tok.${mt_type}.de \
        | cut -d$'\t' -f4 >./pos/test_2016_flickr.lc.norm.tok.${mt_type}.de
    cat ./parzu/test_2016_flickr.lc.norm.tok.${mt_type}.de \
        | cut -d$'\t' -f5 >./stts-pos/test_2016_flickr.lc.norm.tok.${mt_type}.de
done