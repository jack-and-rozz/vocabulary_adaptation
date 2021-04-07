# coding: utf-8
import os, re, sys, random, time, argparse, copy
import glob
sys.path.append(os.getcwd())

from core.utils.common import RED, BLUE, RESET, UNDERLINE

def read_model_outputs(output_path):
  outputs = []
  for l in open(output_path):
    l = l.strip()
    outputs.append(l)
  return outputs



def read_test_data(data_path):
  def separate_dial(line):
    #dial = [[x[:-1] for x in u.split('__eou__')] for u in line.strip().split('__eot__')]
    dial = [' __eou__ '.join(u.split('__eou__')[:-1]) for u in line.strip().split('__eot__')]
    return dial

  contexts = []
  responses = []
  for l in open(data_path):
    dialog = separate_dial(l)
    contexts.append(dialog[:-1])
    responses.append(dialog[-1])
  return contexts, responses

def decorate_unk(text, vocab=None, unk_color=None):
  if not vocab or not unk_color:
    return text

  is_unk = [1 if not w in vocab else 0 for w in text.strip().split()]
  
  return ' '.join([unk_color + w + RESET if not w in vocab else w for w in text.strip().split()]), is_unk

def print_summary(contexts, gold_responses, model_outputs, src_vocab, trg_vocab):
  num_models = len(model_outputs)
  num_unk_in_res = [0 for _ in range(num_models)]
  num_words_in_res = [0 for _ in range(num_models)]
  for i in range(len(contexts)):
    print('[%05d]' % i)
    for j, context in enumerate(contexts[i]):
      # Check the unknown words of the context not in the **trg domain vocab**
      context, _ = decorate_unk(context, vocab=trg_vocab, unk_color=UNDERLINE)
      print('- Context %d' % j, context)
    print()
    print('- Ground Truth:', gold_responses[i])
    for m_idx, (name, outputs) in enumerate(model_outputs.items()):
      # Check the unknown words of the response not in the **src domain vocab**
      output, is_unk = decorate_unk(outputs[i], vocab=src_vocab, unk_color=BLUE)
      num_words_in_res[m_idx] += len(is_unk)
      num_unk_in_res[m_idx] += sum(is_unk)
      print('- Output [%s]:' % name, output)
    print()
  
  print('<Unknown words generation rate>')
  for m_idx, name in enumerate(model_outputs):
    print('[%s]:\t%.5f (%d/%d)' % (name, 1.0 * num_unk_in_res / num_words_in_res, num_unk_in_res, num_words_in_res))

OUTPUT_SUFFIX = '.outputs'

def read_vocab(path, skip_first=True, max_rows=0):
  if max_rows and skip_first:
    max_rows += 1
  vocab = set()
  for i, l in open(path):
    if i ==0 and skip_first:
      continue
    vocab.add(l.strip().split()[0])
  return vocab

def main(args):
  output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename + OUTPUT_SUFFIX))
  output_paths = sorted(output_paths)
  contexts, responses = read_test_data(args.test_file)
  assert len(contexts) == len(responses)
  results = {}
  print('# test file: %d' % len(responses))
  for output_path in output_paths:
    model_name = output_path.split('/')[-3]
    results[model_name] = read_model_outputs(output_path)
    print('# model outputs (%s): %d' % (model_name, len(results[model_name])))

    assert len(results[model_name]) == len(responses)

  src_vocab = read_vocab(args.src_domain_vocab) if args.src_domain_vocab else None
  trg_vocab = read_vocab(args.trg_domain_vocab) if args.trg_domain_vocab else None

  print_summary(contexts, responses, results, src_vocab, trg_vocab)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('models_root', type=str)
  parser.add_argument('test_file', type=str)
  parser.add_argument('--output_filename', default='dialogue_test.best', 
                      type=str)
  parser.add_argument('--src_domain_vocab', default=None, type=str)
  parser.add_argument('--trg_domain_vocab', default=None, type=str)
  
  args = parser.parse_args()
  main(args)
