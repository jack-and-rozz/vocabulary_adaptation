# coding: utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())

import subprocess
from collections import Counter
from common import flatten, dbgprint


def get_shell_var(varname):
    CMD = 'echo $(source const.sh; echo $%s)' % varname
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return p.stdout.readlines()[0].strip().decode('utf-8')


def count_words(path):
    res = set()
    for l in open(path):
        for x in l.strip().split():
            res.add(x)
    # return set(flatten([l.strip().split() for l in open(path)]))
    return res

def main(args):
    src_data_dir = get_shell_var(args.src_domain + '_data_dir')
    tgt_data_dir = get_shell_var(args.tgt_domain + '_data_dir')
    words_input_src = count_words(src_data_dir + '/train.%s' % args.input_lang)
    words_output_src = count_words(src_data_dir + '/train.%s' % args.output_lang)
    words_input_tgt = count_words(tgt_data_dir + '/train.%s' % args.input_lang)
    words_output_tgt = count_words(tgt_data_dir + '/train.%s' % args.output_lang)
    print(src_data_dir + '/train.%s' % args.input_lang)
    print(src_data_dir + '/train.%s' % args.output_lang)
    print(tgt_data_dir + '/train.%s' % args.input_lang)
    print(tgt_data_dir + '/train.%s' % args.output_lang)

    print('# tokens %s(%s)' % (args.src_domain, args.input_lang), len(words_input_src))
    print('# tokens %s(%s)' % (args.src_domain, args.output_lang), len(words_output_src))
    print('# tokens %s(%s)' % (args.tgt_domain, args.input_lang), len(words_input_tgt))
    print('# tokens %s(%s)' % (args.tgt_domain, args.output_lang), len(words_output_tgt))
    print('# shared tokens (%s)' % args.input_lang, len(words_input_src.intersection(words_input_tgt)))
    print('# shared tokens (%s)' % args.output_lang, len(words_output_src.intersection(words_output_tgt)))

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_domain', type=str)
    parser.add_argument('tgt_domain', type=str)
    parser.add_argument('input_lang', default='en', type=str)
    parser.add_argument('output_lang', default='ja', type=str)
    args = parser.parse_args()
    main(args)
