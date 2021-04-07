# coding: utf-8
import os, re, sys, time, argparse, subprocess
# import torch
# from torch.serialization import default_restore_location
from fairseq.utils import import_user_module
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data.dictionary import Dictionary

def main(args):
    import_user_module(args)
    ckpt_path1 = args.model1_root + '/checkpoints/checkpoint_best.pt'
    ckpt_path2 = args.model2_root + '/checkpoints/checkpoint_best.pt'
    state1 = load_checkpoint_to_cpu(ckpt_path1)
    state2 = load_checkpoint_to_cpu(ckpt_path2)

    enc_emb1 = state1['model']['encoder.embed_tokens.weight']
    enc_emb2 = state2['model']['encoder.embed_tokens.weight']
    check = enc_emb1 == enc_emb2

    print(check[:6])
    print(enc_emb1[:6, :5])
    print(enc_emb2[:6, :5])
    # enc_emb = state['model']['encoder.embed_tokens.weight']
    # enc_emb_output_path = args.model_root + '/embeddings/encoder.indomain'
    # output_trained_embeddings_to_file(enc_emb, args.srcdict, 
    #                                   enc_emb_output_path)
    
    # if args.tgtdict:
    #     dec_emb = state['model']['decoder.embed_tokens.weight']
    #     dec_emb_output_path =args.model_root + '/embeddings/decoder.indomain'
    #     output_trained_embeddings_to_file(dec_emb, args.tgtdict, 
    #                                       dec_emb_output_path)

if __name__ == "__main__":
  desc = ''
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('-m1', '--model1_root', 
                      default='checkpoints/latest/jesc_sp16000.outD.all',
                      type=str)
  parser.add_argument('-m2', '--model2_root', 
                      default='checkpoints/latest/jesc_sp16000@aspec_sp16000.va.noft.nova.v_aspec_sp16000_100kmono.llm-idt.nn10.100k',
                      type=str)
  parser.add_argument('--user-dir', default='fairseq/extensions')
  args = parser.parse_args()
  main(args)
