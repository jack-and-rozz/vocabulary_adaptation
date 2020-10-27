# coding: utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())

import subprocess
from collections import Counter
from common import flatten, dbgprint
import pandas as pd

def get_shell_var(varname):
    CMD = 'echo $(source const.sh; echo $%s)' % varname
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return p.stdout.readlines()[0].strip().decode('utf-8')



def unk_wd_rate(data, vocab):
    total_len = 0
    matched_len = 0
    for l in open(data):
        l = l.strip().split()
        total_len += len(l)
        matched_len += len([x for x in l if x in vocab])
    return "%.3f" % (1. - (matched_len / total_len))

unk_spm_rate=unk_wd_rate

def main(args):
    # print(args.para_data)
    # print(args.mono_data)
    # print(args.wd_data)
    # exit(1)
    print(args)
    space_char = "▁"
    para_vocab = set([x.split()[0] for x in open(args.para_vocab_path)])
    mono_vocab = set([x.split()[0] for x in open(args.mono_vocab_path)])
    para_vocab_wd = set([x.split()[0][1:] for x in open(args.para_vocab_path) if x[0] == space_char])
    mono_vocab_wd = set([x.split()[0][1:] for x in open(args.mono_vocab_path) if x[0] == space_char])

    overlap = para_vocab.intersection(mono_vocab)
    print("para-mono overlap rate: ", "%.3f" % (1.0*len(overlap)/len(para_vocab)), "(%d/%d)" % (len(overlap), len(para_vocab)))
    print("word spm rate (para):", "%.3f" % (len(para_vocab_wd)/len(para_vocab)), "(%d/%d)" % (len(para_vocab_wd), len(para_vocab)))
    print("word spm rate (mono):", "%.3f" %  (len(mono_vocab_wd)/len(mono_vocab)), "(%d/%d)" % (len(mono_vocab_wd), len(mono_vocab)))
    
    header=['Model', 'src-train', 'tgt-train.100k', 'tgt-test']

    para_spm_rates = ['para-spm'] +  [unk_spm_rate(data, para_vocab) 
                      for i, data in enumerate(args.para_data)]
    mono_spm_rates = ['mono-spm', '-'] + [unk_spm_rate(data, mono_vocab) 
                                          for i, data in enumerate(args.mono_data)]

    para_wd_rates = ['para-word'] +  [unk_wd_rate(data, para_vocab_wd) 
                                      for i, data in enumerate(args.wd_data)]
    mono_wd_rates = ['mono-word'] +  [unk_wd_rate(data, mono_vocab_wd) 
                                      for i, data in enumerate(args.wd_data)]

    results = [para_spm_rates, mono_spm_rates, para_wd_rates, mono_wd_rates]
    df = pd.DataFrame(results, columns=header).set_index('Model')
    print()
    print(df)



if __name__ == "__main__":
    # Common arguments are defined in base.py
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('para_vocab_path', type=str)
    parser.add_argument('mono_vocab_path', type=str)
    parser.add_argument('--para_data', type=str, nargs='+')
    parser.add_argument('--mono_data', type=str, nargs='+')
    parser.add_argument('--wd_data', type=str, nargs='+')
    # parser.add_argument('system_output', type=str)
    args = parser.parse_args()
    main(args)
