

import os, re, sys, random, time, argparse, copy


def main(args):
    f = open(args.input_emb)
    l = f.readline()
    _, n_dims = l.split()
    n_dims = int(n_dims)
    success_in_prev_step = True
    cnt = 1
    while l:
        cnt += 1
        if success_in_prev_step:
            print(l.strip())
        try:
            l = f.readline()
            success_in_prev_step = True
        except:
            success_in_prev_step = False
            continue
        splited_line = l.rstrip().split(" ")
        if len(splited_line) != n_dims + 1:
            print('line %d' % cnt, splited_line[0], len(splited_line), file=sys.stderr)
            success_in_prev_step = False

if __name__ == "__main__":
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_emb', type=str)
    args = parser.parse_args()
    main(args)
