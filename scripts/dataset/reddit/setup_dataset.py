# coding: utf-8

import os, re, sys, random, time, argparse, copy
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/scripts')
sys.path.append('..')
import pandas as pd
from datetime import datetime

from pprint import pprint
from common import read_jsonlines, separate_path_and_filename, dump_as_json, flatten, dotDict, str2bool, timewatch, recDotDefaultDict


URL1 = re.compile("(\[.*?\])\s*?\((.+?)\)")
URL2 = re.compile("((http(s)?:\/\/)|(www\.))\S+")
#URL3 = re.compile("((<URL>\s*?)|(<URL>, )|(<URL>( __eou__)?)|(<URL>( \|)?)){2,}") # Compress repeated URLs.

SYMURL = "<URL>"
EOU = '__eou__'
EOT = '__eot__'
U_DELIM = ' %s ' % EOU
T_DELIM = ' %s ' % EOT

TYPEIDS = {
  't1': 'Comment',
  't2': 'Account',
  't3':	'Link',
  't4':	'Message',
  't5':	'Subreddit',
  't6':	'Award',
}


def print_sequence(data, id_seq):
  print('***********')
  for i, _id in enumerate(id_seq):
    d = data[_id]
    print(i, d.body)


def filter_dialog(data, id_seq):
  if True in [data[_id].is_deleted for _id in id_seq]:
    return False
  return True

def process_uttr(uttr, lowercase=False, disable_normalize_url=False):
  uttr = [' '.join([x for x in sent.strip().split() if x]) 
          for sent in uttr.split('\n') if sent.strip()]
  uttr = U_DELIM.join(uttr) 
  uttr = uttr.replace('^', '')
  if lowercase:
    uttr = uttr.lower()

  if not disable_normalize_url:
      uttr = URL1.sub(SYMURL, uttr)
      uttr = URL2.sub(SYMURL, uttr)
      #uttr = URL3.sub(SYMURL, uttr)
  return uttr.strip()


def show_body(d):
    if not d.is_processed:
        d.body = process_uttr(d.body, 
                              lowercase=args.lowercase,
                              disable_normalize_url=args.disable_normalize_url)
        d.is_processed = True
    return d.body


@timewatch()
def find_leaves(data):
  assert type(data) == list

  is_leaf = dotDict()
  new_data = dotDict() #recDotDefaultDict() #dotDict()
  for d in data:
    d.is_processed = False # For lazy preprocessing.
    d.is_deleted = True if d.body.strip() == '[deleted]' else False

    d.is_root = True if not d.parent_id else False
    new_data[d.id] = d
    if d.id not in is_leaf:
      is_leaf[d.id] = True

    if d.parent_id:
      parent_id = d.parent_id[3:] # parent_id includes a prefix specifying the mention type in addition to its actual parent's id.
      is_leaf[parent_id] = False
  leaves = [k for k,v in is_leaf.items() if v is True]
  return new_data, leaves


@timewatch()
def trace_from_leaves_to_root(data, leaf_ids):
  def _trace(data, _id, depth=1):
    parent_id = data[_id].parent_id[3:]
    if depth == 10:
      return [_id]
    elif parent_id and parent_id != _id and parent_id in data:
      return _trace(data, parent_id, depth=depth+1) + [_id]
    else:
      return [_id]
  id_seqs = []
  for i, _id in enumerate(leaf_ids):
    id_seq = _trace(data, _id)
    if len(id_seq) >= 2:
      id_seqs.append(id_seq)
  return id_seqs

def get_dialogs(data):
  data, leaf_ids = find_leaves(data)
  id_seqs = trace_from_leaves_to_root(data, leaf_ids)
  id_seqs = [id_seq for id_seq in id_seqs if filter_dialog(data, id_seq)]

  # for id_seq in id_seqs:
  #   print_sequence(data, id_seq)
  return data, id_seqs

def save_data(target_path, data, id_seqs):
    with open(target_path + '.ids', 'w') as f:
        for id_seq in id_seqs:
            f.write(' '.join(id_seq) + '\n')

    with open(target_path + '.all.txt', 'w') as f:
        for id_seq in id_seqs:
            dial = T_DELIM.join([show_body(data[_id]) for _id in id_seq])
            f.write(dial + '\n')

    with open(target_path + '.subreddits', 'w') as f: 
        for id_seq in id_seqs:
            f.write(data[id_seq[-1]]['subreddit'] + '\n')

    with open(target_path + '.authors', 'w') as f: 
        for id_seq in id_seqs:
            f.write(' '.join([data[_id]['author'] for _id in id_seq]) + '\n')
 
@timewatch()
def main(args):
  random.seed(0)
  data = read_jsonlines(args.source_path, max_rows=args.max_rows)
  source_dir, _ = separate_path_and_filename(args.source_path)
  target_dir = args.target_dir if args.target_dir else source_dir
  os.makedirs(target_dir, exist_ok=True)
  target_filenames = [
    'train.all.txt', 'dev.all.txt', 'test.all.txt',
    'train.ids', 'dev.ids', 'test.ids',
  ]

  n_dev = int(len(data) * args.n_dev / 100.0)
  n_test = int(len(data) * args.n_test / 100.0)
  n_train = len(data) - n_dev - n_test

  # Log statistics.
  print('Split the comments into (train, dev, test) = (%d, %d, %d) lines.' \
        % (n_train, n_dev, n_test))

  train_data, train_ids = get_dialogs(data[:n_train])
  dev_data, dev_ids = get_dialogs(data[n_train:n_train+n_dev])
  test_data, test_ids = get_dialogs(data[n_train+n_dev:])

  target_path = target_dir + '/train'
  save_data(target_path, train_data, train_ids)
  target_path = target_dir + '/dev'
  save_data(target_path, dev_data, dev_ids)
  target_path = target_dir + '/test'
  save_data(target_path, test_data, test_ids)


if __name__ == "__main__":
  desc = ""
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-trg', '--target_dir', default=None)
  parser.add_argument('-src', '--source_path', default='dataset/reddit/original/RC_2015-01')
  parser.add_argument('-mr', '--max_rows', type=int, default=0)
  parser.add_argument('-ndv', '--n_dev', type=float, default=2.5,
                      help='Devision ratio for dev data.')
  parser.add_argument('-nts', '--n_test', type=float, default=2.5,
                      help='Devision ratio for test data.')
  parser.add_argument('--disable_normalize_url', action='store_true', 
                      default=False)
  parser.add_argument('--lowercase', action='store_true', default=False)
  parser.add_argument('--overwrite', action='store_true', default=False,
                      help='If set true, and overwrite existing preprocessed dataset.')

  args = parser.parse_args()
  main(args)


