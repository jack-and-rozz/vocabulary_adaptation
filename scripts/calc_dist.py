# coding: utf-8
import os, re, sys, time, argparse, subprocess
import glob
sys.path.append(os.getcwd())

from common import RED, BLUE, RESET, UNDERLINE

def read_data(file_path):
    return [l.strip() for l in open(file_path)]

def calc_dist(hypotheses, dist_n):
  '''
  https://www.aclweb.org/anthology/N16-1014
  '''
  assert dist_n >= 1
  assert type(hypotheses[0]) == str

  n_total_words = 0
  uniq_words = set()
  for hypothesis in hypotheses:
    words_in_hyp = [x for x in hypothesis.split() if x]
    ngrams = [tuple(words_in_hyp[i:i+dist_n]) for i in range(len(words_in_hyp)- dist_n+1)]
    for ngram in ngrams:
      uniq_words.add(ngram)
    n_total_words += len(ngrams)
  return 100.0 * len(uniq_words) / n_total_words if n_total_words else 0


def calc_length(hypotheses):
    lens = [len(l.strip().split()) for l in hypotheses]
    return sum(lens) / len(lens)

def main(args):
    output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename))
    output_paths = sorted(output_paths)
    outputs = {}
    for path in output_paths:
        header = path.split('/')[-3]
        outputs[header] = read_data(path)

    references = read_data(args.reference_file)

    print('<dist>')
    dist1 =  calc_dist(references, 1)
    dist2 =  calc_dist(references, 2)
    average_length = calc_length(references)

    header = 'reference'
    print(header, 'dist-1/2= %.2f, %.2f, average length= %.2f' %(dist1, dist2, average_length))

    for output_path in output_paths:
        #header = '/'.join(output_path.split('/')[:-2])
        header = output_path.split('/')[-3]
        dist1 = calc_dist(outputs[header], 1)
        dist2 = calc_dist(outputs[header], 2)
        average_length = calc_length(outputs[header])
        print(header, 'dist-1/2= %.2f, %.2f, average length= %.2f' %(dist1, dist2,average_length))

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('models_root', type=str)
  parser.add_argument('output_filename', type=str)
  parser.add_argument('reference_file', type=str)
  args = parser.parse_args()
  main(args)
