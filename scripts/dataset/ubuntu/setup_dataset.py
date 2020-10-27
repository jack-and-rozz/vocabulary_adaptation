# coding:utf-8
import sys, os, random, copy, time, re, argparse
sys.path.append(os.getcwd())

import pandas as pd
from tqdm import tqdm

URL = re.compile("((http(s)?:\/\/)|(www\.))\S+")
SYMURL = "<URL>"
DELIM = " __eot__ "

# def process_uttr(uttr):
#     uttr = URL.sub(SYMURL, uttr)
#     return ' '.join(uttr.strip().split())

EOU = '__eou__'
EOT = '__eot__'
U_DELIM = ' %s ' % EOU
T_DELIM = ' %s ' % EOT

def get_first_uttr(uttr):
    return uttr.split(EOU)[0].strip()

def process_data(df, file_header, num_turns, include_all_splits=False):
    short_src = open(args.source_dir + '/%s.%dturn.src' % (file_header, num_turns), 'w')
    short_tgt = open(args.source_dir + '/%s.%dturn.tgt' % (file_header, num_turns), 'w')
    for index, (context, response) in df.iterrows():
        # Replace URLs and separate a context into turns.
        context = [' '.join(turn.strip().split()) for turn in URL.sub(SYMURL, context).strip().split(EOT) if turn.strip()]
        # Separate a turn into uttrs.
        context = [U_DELIM.join([uttr for uttr in turn.split(EOU) if uttr.strip()]) for turn in context]

        # Separate a turn into uttrs.
        response = U_DELIM.join([uttr.strip() for uttr in URL.sub(SYMURL, response).strip().split(EOU) if uttr.strip()])

        dialog = context + [response]

        # DEBUG
        # dialog = list(map(str, [1,2,3,4]))
        # dialog = [x + ' __eou__ ' + x for x in dialog]

        print(T_DELIM.join(dialog[-(num_turns+1):-1]), file=short_src)
        print(get_first_uttr(dialog[-1]), file=short_tgt)

        if include_all_splits:
            '''
            Each split contains up to args.max_turn+1 utterances.
            e.g.) dialog = [1,2,3,4], max_turn=2
            -> [2,3,4], [1,2,3], [1,2]
            '''
            for split in [dialog[:-i-1] for i in range(len(dialog) - 2)]:
                # print(T_DELIM.join(split[-num_turns:-1]), split[-1])
                print(T_DELIM.join(split[-(num_turns+1):-1]), file=short_src)
                print(get_first_uttr(split[-1]), file=short_tgt)
        # return # debug


def main(args):
    N = args.max_rows if args.max_rows else None

    # Load original files.
    train = pd.read_csv(args.source_dir + '/train.csv', nrows=N)
    train = train.loc[train['Label'] == 1.0].loc[:, ['Context', 'Utterance']]
    dev = pd.read_csv(args.source_dir + '/dev.csv', nrows=N) \
          .loc[:, ['Context', 'Ground Truth Utterance']]

    test = pd.read_csv(args.source_dir + '/test.csv', nrows=N) \
             .loc[:, ['Context', 'Ground Truth Utterance']]
    process_data(train, 'train', args.num_turns, True)
    process_data(dev, 'dev', args.num_turns, False)
    process_data(test, 'test', args.num_turns, False)

if __name__ == "__main__":
    # Common arguments are defined in base.py
    desc = ""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--source_dir', type=str,
                        default='dataset/ubuntudialog/original')
    parser.add_argument('-mr', '--max_rows', type=int, default=0, 
                        help='for debug.')
    parser.add_argument('--num-turns', type=int, default=3, 
                        help='The maximum number of utterances in a conversations, including the response.')
    args = parser.parse_args()
    main(args)
  
