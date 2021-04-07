# coding: utf-8
import os, re, sys, time, argparse, subprocess
# import torch
# from torch.serialization import default_restore_location
from fairseq.utils import import_user_module
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data.dictionary import Dictionary

def load_dict(path):
    return [l.strip()[0].split() for l in open(path)]

def save_emb(path, emb):
    pass


def output_trained_embeddings_to_file(emb, dict_path, tgt_path):
    emb_dict = Dictionary.load(dict_path)
    emb = emb.data
    with open(tgt_path, 'w') as f:
        sys.stdout = f
        print(emb.shape[0], emb.shape[1])
        for i in range(emb.shape[0]):
            print(emb_dict.symbols[i], ' '.join(['%f' % x for x in emb[i]]))
        sys.stdout = sys.__stdout__

def main(args):
    import_user_module(args)
    ckpt_path = args.model_root + '/checkpoints/checkpoint_best.pt'
    # state = torch.load(
    #     ckpt_path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
    # )
    state = load_checkpoint_to_cpu(ckpt_path)

    enc_emb = state['model']['encoder.embed_tokens.weight']
    enc_emb_output_path = args.model_root + '/embeddings/encoder.indomain'
    output_trained_embeddings_to_file(enc_emb, args.srcdict, 
                                      enc_emb_output_path)
    
    if args.tgtdict:
        dec_emb = state['model']['decoder.embed_tokens.weight']
        dec_emb_output_path = args.model_root + '/embeddings/decoder.indomain'
        output_trained_embeddings_to_file(dec_emb, args.tgtdict, 
                                          dec_emb_output_path)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('model_root', type=str)
  parser.add_argument('srcdict', type=str)
  parser.add_argument('--tgtdict', type=str, default='')
  parser.add_argument('--user-dir', default=None)
  args = parser.parse_args()
  main(args)
