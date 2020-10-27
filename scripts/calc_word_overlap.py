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

# def read_dict(path, max_rows=None):
#     vocab = [l.split(' ')[0] for i, l in enumerate(open(path)) 
#              if not max_rows or i < max_rows]
#     return set(vocab)

def read_text(path):
    return [l.strip().split() for l in open(path)]


def read_vocab_from_train(path):
    return set(flatten([l.strip().split() for l in open(path)]))

def stat(lang):
    src_data_dir = get_shell_var(args.src_domain + '_data_dir')
    tgt_data_dir = get_shell_var(args.tgt_domain + '_data_dir')

    # src_input_vocab = read_dict(src_data_dir + '/dict.%s.txt' % lang)
    # tgt_input_vocab = read_dict(tgt_data_dir + '/dict.%s.txt' % lang)
    # src_input_vocab = read_dict(src_data_dir + '/dict.%s.txt' % lang)
    # tgt_input_vocab = read_dict(tgt_data_dir + '/dict.%s.txt' % lang)

    src_input_vocab = read_vocab_from_train(src_data_dir + '/train.%s' % lang)
    tgt_input_vocab = read_vocab_from_train(tgt_data_dir + '/train.%s' % lang)

    shared_input_vocab = src_input_vocab.intersection(tgt_input_vocab)
    print("# (src, tgt, shared) lexicons (%s): " % lang, len(src_input_vocab), len(tgt_input_vocab), len(shared_input_vocab))


    # test_inputs = read_text(tgt_data_dir + '/test.%s' % (lang))
    # test_outputs = None
    # if os.path.exists(tgt_data_dir + '/test.tgt'):
    #     test_outputs = read_text(tgt_data_dir + '/test.tgt')

    # print('# lexicons in test inputs', len(set(flatten(test_inputs))))
    # if test_outputs:
    #     print('# lexicons in test outputs', len(set(flatten(test_outputs))))

    # print(len(shared_input_vocab.intersection(words_in_test_inputs)))
    # for w in shared_input_vocab.intersection(words_in_test_inputs):
    #     print(w)

def main(args):
    stat(args.input_lang)
    if args.output_lang:
        stat(args.output_lang)

if __name__ == "__main__":
    # Common arguments are defined in base.py
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_domain', type=str)
    parser.add_argument('tgt_domain', type=str)
    parser.add_argument('-il', '--input_lang', default='en', type=str)
    parser.add_argument('-ol', '--output_lang', default=None, type=str)
    # parser.add_argument('system_output', type=str)
    args = parser.parse_args()
    main(args)

