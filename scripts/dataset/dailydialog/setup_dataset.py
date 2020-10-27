# coding: utf-8
import os, re, sys, random, time, argparse, copy
import glob
from itertools import chain
from pprint import pprint

delim = '__eou__'

def add_space(delim):
    return ' ' + delim + ' '

def flatten(l):
    return list(chain.from_iterable(l))

def decompose(dialogs):
    return flatten([[d[:i] for i in range(2, len(d))] + [d] for d in dialogs])

def get_last_n_turns(dialogs, num_turns):
    return [d[-(num_turns+1):] for d in dialogs]

def save(dialogs, path_prefix):
    cf = open(path_prefix + '.s', 'w')
    rf = open(path_prefix + '.t', 'w')
    for d in dialogs:
        print(add_space(delim).join(d[:-1]), file=cf)
        print(d[-1], file=rf)

def main(args):
    suffix = '/processed.%dturn' % args.num_turns
    target_dir = args.source_dir + suffix
    os.makedirs(target_dir, exist_ok=True)

    train_path = os.path.join(args.source_dir, args.train)
    dev_path = os.path.join(args.source_dir, args.dev)
    test_path = os.path.join(args.source_dir, args.test)

    train = [[u.strip() for u in l.strip().split(delim)[:-1]] 
             for l in open(train_path)]
    dev = [[u.strip() for u in l.strip().split(delim)[:-1]] 
           for l in open(dev_path)]
    test = [[u.strip() for u in l.strip().split(delim)[:-1]] 
            for l in open(test_path)]
    train = get_last_n_turns(decompose(train), args.num_turns)
    dev = get_last_n_turns(dev, args.num_turns)
    test = get_last_n_turns(test, args.num_turns)

    save(train, target_dir + '/train')
    save(dev, target_dir + '/dev')
    save(test, target_dir + '/test')

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--source-dir', 
                        default='dataset/dailydialog/ijcnlp_dailydialog', type=str)
    parser.add_argument('--train', default='train/dialogues_train.txt')
    parser.add_argument('--dev', default='validation/dialogues_validation.txt')
    parser.add_argument('--test', default='test/dialogues_test.txt')
    parser.add_argument('-nt', '--num-turns', default=1, type=int)
    args = parser.parse_args()
    main(args)
