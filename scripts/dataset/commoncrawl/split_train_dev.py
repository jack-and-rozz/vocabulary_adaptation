

import argparse, random, sys


def read_file(path):
    return [l.strip() for l in open(path)]
    pass

def main(args):
    random.seed(0)
    output_dir = '/'.join(args.src_lang_archive.split('/')[:-1])
    src = read_file(args.src_lang_archive)
    tgt = read_file(args.tgt_lang_archive)
    train_indice = set(range(len(src)))
    dev_indice = set(random.sample(train_indice, args.ndev))
    train_indice -= dev_indice

    with open(output_dir + '/train.%s' % args.src_lang, 'w') as f:
        for idx in train_indice:
            print(src[idx], file=f)
    with open(output_dir + '/train.%s' % args.tgt_lang, 'w') as f:
        for idx in train_indice:
            print(tgt[idx], file=f)

    with open(output_dir + '/dev.%s' % args.src_lang, 'w') as f:
        for idx in dev_indice:
            print(src[idx], file=f)
    with open(output_dir + '/dev.%s' % args.tgt_lang, 'w') as f:
        for idx in dev_indice:
            print(tgt[idx], file=f)


if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_lang_archive', type=str)
    parser.add_argument('tgt_lang_archive', type=str)
    parser.add_argument('--ndev', default=2000, type=int)
    parser.add_argument('--src_lang', default='en', type=str)
    parser.add_argument('--tgt_lang', default='de', type=str)

    # parser.add_argument('tgt_file', type=str)
    # parser.add_argument('N', type=str)
    args = parser.parse_args()
    main(args)
