
import os, re, sys, random, time, argparse, copy



EOT = '__eot__'
EOU = '__eou__'
T_DELIM = ' %s ' % EOT

def main(args):
    src_output = open(args.src_output, 'w')
    tgt_output = open(args.tgt_output, 'w')

    for nline, l in enumerate(sys.stdin):
        if args.max_rows and nline >= args.max_rows:
            break

        dialog = l.strip().split(EOT)
        for split in [dialog[:-i-1] for i in range(len(dialog) - 2)]:
            context = T_DELIM.join(split[-args.max_turns:-1])
            response = split[-1]
            if not args.max_tokens or (len(context.split()) <= args.max_tokens and \
                                       len(response.split()) <= args.max_tokens):
                print(context, file=src_output)
                print(response, file=tgt_output)

        context = T_DELIM.join(dialog[-args.max_turns:-1])
        response = dialog[-1]
        if not args.max_tokens or (len(context.split()) <= args.max_tokens and \
                                   len(response.split()) <= args.max_tokens):
            print(context, file=src_output)
            print(response, file=tgt_output)


if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('src_output', type=str)
  parser.add_argument('tgt_output', type=str)
  parser.add_argument('--max-turns', type=int, default=3)
  parser.add_argument('--max-tokens', type=int, default=70)
  parser.add_argument('-mr', '--max_rows', type=int, default=0, 
                      help='for debug.')

  args = parser.parse_args()
  main(args)


