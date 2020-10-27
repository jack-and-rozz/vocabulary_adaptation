# coding: utf-8
import argparse, random, sys, os, subprocess
import glob
from collections import defaultdict
import random

def main(args):
    file_dir = '/'.join(args.source_path.split('/')[:-1])
    file_name = args.source_path.split('/')[-1].split('.')[0]
    header = file_dir + '/' + file_name + '.%dturn' % args.num_turns
    N = args.num_turns
    delim = args.delimiter

    src_output_file = open(header + '.src', 'w')
    tgt_output_file = open(header + '.tgt', 'w')

    sys.stderr.write("Output separated files to '%s.*'.\n" % header)
    for l in open(args.source_path):
        dialog = [x.strip() for x in l.strip().split(args.delimiter)]

        if len(dialog) >= N + 1:
            uttr = delim.join(dialog[:N])
            res = dialog[N]
        else:
            uttr = delim.join(dialog[:-1])
            res = dialog[-1]

        src_output_file.write(uttr + '\n')
        tgt_output_file.write(res + '\n')

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('source_path')
    parser.add_argument('-nt', '--num-turns', default=1, type=int)
    parser.add_argument('-delim', '--delimiter', default='<EOT>')
    args = parser.parse_args()
    main(args)
