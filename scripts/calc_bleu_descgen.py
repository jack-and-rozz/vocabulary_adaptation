import argparse, random, sys
import subprocess
import glob
from collections import defaultdict
import pandas as pd
def calc_sent_bleu(hyp_filepath, ref_filepath, bleu_path):
    with open(hyp_filepath, 'r') as hyp_f:
        bleu_cmd = [bleu_path, ref_filepath] #+ ['<', ref_filepath]
        # bleu_cmd = './'+ bleu_path
        bleu_out = subprocess.check_output(bleu_cmd, stdin=hyp_f, stderr=sys.stderr)
        
                                           #stderr=subprocess.STDE)
        bleu_out = bleu_out.decode('utf-8').split('\n')
    return [100.0 * float(v) for v in bleu_out if v]

def gather_indice_to_same_word(path):
    same_word_indice = defaultdict(list)
    for i, l in enumerate(open(path)):
        word = l.split('%')[0]
        same_word_indice[word].append(i)
    return same_word_indice

def calc_norbleu(bleu_list, same_word_indice):
    bleus_by_word = defaultdict(list)
    for word, indice in same_word_indice.items():
        bleus = [bleu_list[idx] for idx in indice]
        bleus_by_word[word] = sum(bleus) / len(bleus)
    norbleu = list(bleus_by_word.values())
    norbleu = sum(norbleu) / len(norbleu)
    return norbleu

def main(args):
    print()
    #print(args)
    print(args, file=sys.stderr)
    output_paths = glob.glob("%s/*/tests/%s" % (args.models_root, args.output_filename))
    output_paths = sorted(output_paths)
    model_names = [p.split('/')[-3] for p in output_paths]
    
    same_word_indice = gather_indice_to_same_word(args.word_list) if args.word_list else None

    bleu_avg_all = []
    bleu_avg_by_word = []

    for hyp_path in output_paths:
        res = calc_sent_bleu(hyp_path, 
                             args.reference_path, 
                             args.sentence_bleu_path)

        bleu_avg_all.append(sum(res)/len(res))
        if args.word_list:
            bleu_avg_by_word.append(calc_norbleu(res, same_word_indice))

    if args.word_list:
        bleu_avg_by_word = ["%.2f" % x for x in bleu_avg_by_word]
        header = ['Model', 'Avg-by-word']
        data = [model_names, bleu_avg_by_word]
    else:
        bleu_avg_all = ["%.2f" % x for x in bleu_avg_all]
        header = ['Model', 'Avg-all']
        data = [model_names, bleu_avg_all]

    print('<sentence-BLEU>')
    for l in list(zip(*data)):
        print(' '.join([str(x) for x in l]))

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('models_root', type=str)
    parser.add_argument('output_filename', type=str)
    parser.add_argument('reference_path', type=str)
    parser.add_argument('--word_list', default='', type=str)
    parser.add_argument('--sentence_bleu_path', default='scripts/sentence-bleu')
  
    parser.add_argument
    args = parser.parse_args()
    main(args)
