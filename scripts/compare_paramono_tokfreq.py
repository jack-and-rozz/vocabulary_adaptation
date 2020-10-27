# coding: utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())
from pprint import pprint
import subprocess
from collections import Counter, defaultdict
from common import flatten, dbgprint, timewatch
import pandas as pd
import matplotlib.pyplot as plt

UNK = '<unk>'


def hist(data, vocab):
    hist = {key:0 for key in vocab}
    hist[UNK] = 0
    cnt = Counter(flatten(data))
    for k, v in cnt.items():
        if k in hist:
            hist[k] += v
        else:
            hist[UNK] += v
    return hist


def draw(hist1, hist2):
    left1 = list(range(len(hist1)))
    height1= list(sorted(hist1.values(), reverse=True))
    left2 = list(range(len(hist2)))
    height2= list(sorted(hist2.values(), reverse=True))


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(left1, height1, width=1.0, alpha=0.5, color='red', log=True)
    ax.bar(left2, height2, width=1.0, alpha=0.5, color='blue', log=True)


    # fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(20, 8), squeeze=True)
    # axes[0].bar(left1, height1, width=1.0, alpha=0.5, color='red', log=True)
    # axes[1].bar(left2, height2, width=1.0, alpha=0.5, color='blue', log=True)

    plt.savefig('paramono.pdf', bbox_inches="tight", pad_inches=0.03)



def main(args):
    print('para-src', args.para_src_data_dir)
    print('para-tgt', args.para_tgt_data_dir)
    print('mono-src', args.mono_src_data_dir)
    print('mono-tgt', args.mono_tgt_data_dir)

    para_sdom_vocab = set([x.split()[0] for x in open(args.para_src_data_dir + '/dict.%s.txt' % args.lang)])
    mono_sdom_vocab = set([x.split()[0] for x in open(args.mono_src_data_dir + '/dict.%s.txt' % args.lang)])

    para_tdom_vocab = set([x.split()[0] for x in open(args.para_tgt_data_dir + '/dict.%s.txt' % args.lang)])
    mono_tdom_vocab = set([x.split()[0] for x in open(args.mono_tgt_data_dir + '/dict.%s.txt' % args.lang)])

    # print(len(para_sdom_vocab.intersection(para_tdom_vocab)))
    # print(len(mono_sdom_vocab.intersection(mono_tdom_vocab)))
    # exit(1)

    para_sdom_train = [l.strip().split() for l in open(args.para_src_data_dir + '/train.all.%s' % args.lang)]
    mono_sdom_train = [l.strip().split() for l in open(args.mono_src_data_dir + '/train.all.%s' % args.lang)]

    para_tdom_train = [l.strip().split() for l in open(args.para_tgt_data_dir + '/train.100k.%s' % args.lang)]
    mono_tdom_train = [l.strip().split() for l in open(args.mono_tgt_data_dir + '/train.100k.%s' % args.lang)]
    para_tdom_test =  [l.strip().split() for l in open(args.para_tgt_data_dir + '/test.%s' % args.lang)]
    mono_tdom_test =  [l.strip().split() for l in open(args.para_tgt_data_dir + '/test.%s' % args.lang)]
    
    para_tdom_ave_len = len(flatten(para_tdom_train)) / len(para_tdom_train)
    mono_tdom_ave_len = len(flatten(mono_tdom_train)) / len(mono_tdom_train)

    print('Average tok/sent (para): ', para_tdom_ave_len)
    print('Average tok/sent (mono): ', mono_tdom_ave_len)

    para_sdom_hist = hist(para_sdom_train, para_sdom_vocab)
    mono_sdom_hist = hist(mono_sdom_train, mono_sdom_vocab)
    
    para_tdom_hist = hist(para_tdom_train, para_tdom_vocab)
    mono_tdom_hist = hist(mono_tdom_train, mono_tdom_vocab)

    # pprint(sorted(list(para_sdom_hist.items()), key=lambda x: -x[1])[:100])

    thresholds = [0, 1, 3, 5, 10, 20, 100, 1000, 10000]

    print('<Infrequent subwords in src domain>')
    for t in thresholds:
        para = [(k, v) for k, v in para_sdom_hist.items() if v <= t]
        mono = [(k, v) for k, v in mono_sdom_hist.items() if v <= t]
        print('(freq <=%d in src-trainAll) (para, mono) =' % t, len(para), len(mono))

    print('<Infrequent subwords in tgt domain>')
    for t in thresholds:
        para = [(k, v) for k, v in para_tdom_hist.items() if v <= t]
        mono = [(k, v) for k, v in mono_tdom_hist.items() if v <= t]
        print('(freq <=%d in tgt-train100k) (para, mono) =' % t, len(para), len(mono))
    para_both_hist = defaultdict(int)
    for k, v in para_sdom_hist.items():
        para_both_hist[k] += v
    for k, v in para_tdom_hist.items():
        para_both_hist[k] += v

    mono_both_hist = defaultdict(int)
    for k, v in mono_sdom_hist.items():
        mono_both_hist[k] += v
    for k, v in mono_tdom_hist.items():
        mono_both_hist[k] += v


    print('<Infrequent subwords in both domains>')
    for t in thresholds:
        para = [(k, v) for k, v in para_both_hist.items() if v <= t]
        mono = [(k, v) for k, v in mono_both_hist.items() if v <= t]
        print('(freq <=%d in tgt-train100k) (para, mono) =' % t, len(para), len(mono))

    # print('<para>')
    # print(para)

    # print('<mono>')
    # print(mono)
    # print()
    # print(len(para), len(mono))
    # draw(para_tdom_hist, mono_tdom_hist)

if __name__ == "__main__":
    # Common arguments are defined in base.py
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('para_src_data_dir', type=str)
    parser.add_argument('para_tgt_data_dir', type=str)
    parser.add_argument('mono_src_data_dir', type=str)
    parser.add_argument('mono_tgt_data_dir', type=str)
    parser.add_argument('lang', type=str)
    # parser.add_argument('system_output', type=str)
    args = parser.parse_args()
    main(args)
