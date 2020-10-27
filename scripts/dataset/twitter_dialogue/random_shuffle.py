# coding: utf-8
import argparse, random, sys, os, subprocess
import glob
from collections import defaultdict
import random
random.seed(0)

def load(path):
    return [l for l in open(path)]

def main(args):
    os.makedirs(args.tgt_dir, exist_ok=True)
    src_header = args.src_dir + '/' + args.header
    tgt_header = args.tgt_dir + '/' + args.header
    n_examples = subprocess.getoutput('wc -l ' + src_header + args.suffix[0])
    n_examples = int(n_examples.split()[0])
    shuffled_idx = list(range(n_examples))
    random.shuffle(shuffled_idx) 

    for suf in args.suffix:
        data = load(src_header + suf)
        with open(tgt_header + suf, 'w') as f:
            for idx in shuffled_idx:
                f.write(data[idx])

    with open(tgt_header + '.idx', 'w') as f:
        for idx in shuffled_idx:
            f.write('%d\n' % idx)

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('header')
    parser.add_argument('-src', '--src_dir', 
                        default='dataset/twitter/tokenized.mecab')
    parser.add_argument('-tgt', '--tgt_dir', 
                        default='dataset/twitter/tokenized.mecab.shuf')
    parser.add_argument('-suf', "--suffix", nargs="+", type=str,
                        default=['.dialogs', '.tids', '.uids', '.utime'])
    args = parser.parse_args()
    main(args)
