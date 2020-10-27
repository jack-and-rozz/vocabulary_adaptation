# coding:utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())


def main(args):
    # src_file = args.src_file.split('.')
    # filtered_src_name = '.'.join(src_file[:-1]) + '.filtered.' + src_file[-1]
    # tgt_file = args.tgt_file.split('.')
    # filtered_tgt_name = '.'.join(tgt_file[:-1]) + '.filtered.' + tgt_file[-1]
    # filtered_src = open(filtered_src_name, 'w')
    # filtered_tgt = open(filtered_tgt_name, 'w')
    # for context, response in zip(open(args.src_file), open(args.tgt_file)):
    #     context = context.strip()
    #     response = response.strip()
    #     if len(context.split()) <= args.max_tokens and len(response.split()) <= args.max_tokens:
    #         print(context, file=filtered_src)
    #         print(response, file=filtered_tgt)


if __name__ == "__main__":
    # Common arguments are defined in base.py
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_file', type=str)
    parser.add_argument('tgt_file', type=str)
    parser.add_argument('--max-tokens', type=int, default=70, 
                        help='The maximum number of tokens in a context or a response.')
    args = parser.parse_args()
    main(args)
