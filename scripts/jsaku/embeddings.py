import sys
import logging
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
import itertools

import click
import numpy as np
import pandas as pd

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BLACK = "\033[30m"
UNDERLINE = '\033[4m'
BOLD = BLACK+"\033[1m" 
RESET = "\033[0m"


logger = logging.getLogger(__name__)

def load_emb(fobj, n_vocab=None, candidate_words=None):
    _n_vocab, dim = map(int, fobj.readline().split())

    n_vocab = n_vocab or _n_vocab

    if candidate_words:
        n_vocab = min(n_vocab, len(candidate_words))

    vecs = np.empty((n_vocab, dim), dtype=np.float32)
    words = []

    for i, line in enumerate(fobj):
        if len(words) >= n_vocab:
            break
        word, vec_str = line.split(' ', 1)
        if candidate_words and word not in candidate_words:
            continue
        vecs[len(words)] = np.fromstring(vec_str, sep=' ')
        words.append(word)
    return words, vecs

def save_emb(fobj, words, vecs):
    print("{} {}".format(len(words), len(vecs[0])), file=fobj)

    for i, word in enumerate(words):
        vec_str = ' '.join('{:.6f}'.format(v) for v in vecs[i])
        print('{} {}'.format(word, vec_str), file=fobj)

def normalize_emb(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms==0] = 1
    return vecs / norms[:, None]

def get_word2id(words):
    return {word: idx for idx, word in enumerate(words)}

@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@cli.command()
@click.argument('src-emb-path', type=click.Path(exists=True))
@click.argument('tgt-emb-path', type=click.Path(exists=True))
@click.option('--n-vocab', type=int)
def listup_neighbor_candidates(src_emb_path, tgt_emb_path, n_vocab):
    with open(src_emb_path) as f:
        src_words, src_vecs = load_emb(f, n_vocab)
    with open(tgt_emb_path) as f:
        tgt_words, tgt_vecs = load_emb(f, n_vocab)
        tgt_words = set(tgt_words)

    for word in src_words:
        if word in tgt_words:
            print(word)

@cli.command()
@click.argument('src-emb-path', type=click.Path(exists=True))
@click.argument('tgt-emb-path', type=click.Path(exists=True))
@click.option('--candidates-vocab-path', type=str, default=None)
@click.option('--k', type=int, default=10)
@click.option('--n-vocab', type=int)
@click.option('--score', default=False)
@click.option('--format', type=click.Choice(['plain', 'json']), default='plain')
@click.option('--stdin', is_flag=True)
def nn(src_emb_path, tgt_emb_path, candidates_vocab_path, 
       k, n_vocab, score, format, stdin):
    
    neighbor_candidates = set([l.rstrip() for l in open(candidates_vocab_path)])   if candidates_vocab_path else None

    logger.info("Loading embeddings from %s ..." % src_emb_path)
    with open(src_emb_path) as f:
        src_words, src_vecs = load_emb(f, n_vocab)
        src_word2id = get_word2id(src_words)
        src_vecs = normalize_emb(src_vecs)

    logger.info("Loading embeddings from %s ..." % tgt_emb_path)
    with open(tgt_emb_path) as f:
        tgt_words, tgt_vecs = load_emb(f, n_vocab, neighbor_candidates)
        tgt_word2id = get_word2id(tgt_words)
        tgt_vecs = normalize_emb(tgt_vecs)

    def iter_words(words):
        if stdin:
            for line in sys.stdin:
                yield line.strip()
        else:
            for word in words:
                yield word

    for word in iter_words(src_words):
        if word not in src_word2id:
            logger.info("'{}' not in vocab".format(word))
            continue

        src_idx = src_word2id[word]
        src_vec = src_vecs[src_idx]

        cosine = tgt_vecs.dot(src_vec)
        if word in tgt_word2id:
            tgt_idx = tgt_word2id[word]
            cosine[tgt_idx] = -100

        rankings = np.argsort(cosine)[::-1]

        results = dict(src_word=word)

        results['rank'] = [dict(word=tgt_words[tgt_idx], score=float(cosine[tgt_idx])) for tgt_idx in rankings[:k]]

        if format == 'json':
            print(json.dumps(results))
        elif format == 'plain':
            # rank_str = ' '.join('{}/{:.3f}'.format(rank['word'], rank['score']) for rank in results['rank'])
            rank_str = ' '.join('{}'.format(rank['word']) for rank in results['rank'])
            print("{}\t{}".format(results['src_word'], rank_str))

@cli.command()
@click.argument('nn-paths', type=click.Path(exists=True), nargs=-1)
def merge_nn(nn_paths):
    nns = defaultdict(dict)
    for nn_path in nn_paths:
        nns[nn_path] = {}
        with open(nn_path) as f:
            for line in f:
                nn = json.loads(line)
                nns[nn['src_word']][str(nn_path)] = nn['rank']

    for word, ranks in nns.items():
        print(json.dumps(dict(src_word=word, ranks=ranks)))


def load_neighbors(path, format):
    if format == 'plain':
        neighbors = {l.split('\t')[0]:l.split('\t')[1].strip().split(' ') for l in open(path)}
    elif format == 'json':
        raise NotImplementedError
    return neighbors


def compare_overlap(cbow_neighbors, cross_domain_neighbors, 
                    tokens=None, k=None):

    overlaps = {}
    if not tokens:
        tokens = cbow_neighbors.keys()

    for tok in tokens:
        cbow_nn = cbow_neighbors[tok]
        cross_nn = cross_domain_neighbors[tok]
        if k:
            cbow_nn = cbow_nn[:k]
            cross_nn = cross_nn[:k]
        overlaps[tok] = {}
        overlaps[tok]['both'] = set(cbow_nn).intersection(set(cross_nn))
        overlaps[tok]['all'] = [BLUE + c + RESET if c in cbow_nn else RED + c + RESET for c in cross_nn]

        # overlaps[tok]['both'] = cbow_nn.intersection(cross_nn)
        # overlaps[tok]['only_cbow'] = cbow_nn - overlaps[tok]['both']
        # overlaps[tok]['only_cross'] = cross_nn - overlaps[tok]['both']
    overlaps = OrderedDict(sorted(overlaps.items(), 
                                  key=lambda x: -len(x[1]['both'])))
    return overlaps

# @cli.command()
# @click.argument('tgt-cbow-neighbors-path', type=click.Path(exists=True))
# @click.argument('cross-domain-neighbors-path', type=click.Path(exists=True))
# @click.argument('src-nmt-neighbors-path', type=click.Path(exists=True))
# @click.option('--k', type=int, default=10)
# @click.option('--format', type=click.Choice(['plain', 'json']), default='plain')
def nn_overlap(tgt_cbow_neighbors_path, 
               cross_domain_neighbors_path, src_nmt_neighbors_path,
               k, format):

    # Neighbors computed in the tgt-domain CBoW space.
    tgt_cbow_neighbors = load_neighbors(tgt_cbow_neighbors_path, format)

    # Cross-space Neighbors computed in the src-domain NMT space between the transformed CBoW vectors and tgt-domain NMT embeddings.
    cross_domain_neighbors = load_neighbors(cross_domain_neighbors_path, format)

    # Neighbors computed in the src-domain NMT space.
    src_nmt_neighbors = load_neighbors(src_nmt_neighbors_path, format)


    shared_tokens = set(tgt_cbow_neighbors.keys()).intersection(src_nmt_neighbors.keys())
    only_tgt_tokens = set(tgt_cbow_neighbors.keys()) - shared_tokens

    shared_overlaps = compare_overlap(tgt_cbow_neighbors, 
                                      cross_domain_neighbors, 
                                      shared_tokens, k=k)
    tgt_cbow_overlaps = compare_overlap(tgt_cbow_neighbors, 
                                   cross_domain_neighbors, 
                                   only_tgt_tokens, k=k)
    all_overlaps =  compare_overlap(tgt_cbow_neighbors, 
                                    cross_domain_neighbors, k=k)

    def avg_overlaps(overlaps):
        return sum([len(x['both']) for x in overlaps.values()]) / len(overlaps)


    def decorate(k, v, shared_tokens):
        if k not in shared_tokens:
            # k = UNDERLINE + k + RESET
            k = GREEN + k + RESET
        # both = [BLUE + w + RESET for w in v['both']]
        # only_cross = [RED+  w + RESET for w in v['only_cross']]
        # neighbors = both + only_cross
        neighbors = v['all']
        return "%s\t%s" % (k, ', '.join(neighbors))

    # for k, v in all_overlaps.items():
    #     print(decorate(k, v, shared_tokens))

    shared_outputs = OrderedDict()
    tgt_only_outputs = OrderedDict()
    for k, v in tgt_cbow_overlaps.items():
        tgt_only_outputs[k] = decorate(k, v, shared_tokens)

    for k, v in shared_overlaps.items():
        shared_outputs[k] = decorate(k, v, shared_tokens)

    stats = [avg_overlaps(shared_overlaps), avg_overlaps(tgt_cbow_overlaps), avg_overlaps(all_overlaps)]
    return shared_outputs, tgt_only_outputs, stats

    print()
    print('# Avg. overlaps over the shared tokens:', avg_overlaps(shared_overlaps))
    print('# Avg. overlaps over the tokens only in the CBoW space:', avg_overlaps(tgt_cbow_overlaps))

    print('# Avg. overlaps over all tokens:', avg_overlaps(all_overlaps))

@cli.command()
@click.argument('tgt-cbow-neighbors-path', type=click.Path(exists=True))
# @click.argument('cross-domain-neighbors-path1', type=click.Path(exists=True))
@click.argument('cross-domain-neighbors-paths', type=click.Path(exists=True), 
                nargs=2)
@click.argument('src-nmt-neighbors-path', type=click.Path(exists=True))
@click.option('--k', type=int, default=10)
@click.option('--format', type=click.Choice(['plain', 'json']), default='plain')
def nn_overlaps(tgt_cbow_neighbors_path, 
                cross_domain_neighbors_paths, 
                src_nmt_neighbors_path,
                k, format):
    shared_outputs_cbow, tgt_only_outputs_cbow, _ = nn_overlap(
        tgt_cbow_neighbors_path,
        tgt_cbow_neighbors_path, 
        src_nmt_neighbors_path,
        k, format)
    shared_outputs_cross1, tgt_only_outputs_cross1, stat1 = nn_overlap(
        tgt_cbow_neighbors_path, 
        cross_domain_neighbors_paths[0], 
        src_nmt_neighbors_path,
        k, format)
    shared_outputs_cross2, tgt_only_outputs_cross2, stat2 = nn_overlap(
        tgt_cbow_neighbors_path, 
        cross_domain_neighbors_paths[1], 
        src_nmt_neighbors_path,
        k, format)

    for tok in shared_outputs_cross2:
        _tok = shared_outputs_cbow[tok].split('\t')[0]
        print('<%s>: (shared)' % tok)
        print('CBoW  :', '\t'.join(shared_outputs_cbow[tok].split('\t')[1:]))
        print('Linear:', '\t'.join(shared_outputs_cross1[tok].split('\t')[1:]))
        print('LLM   :', '\t'.join(shared_outputs_cross2[tok].split('\t')[1:]))
        print()

    sys.stdout = sys.stderr
    for tok in tgt_only_outputs_cross2:
        _tok = tgt_only_outputs_cbow[tok].split('\t')[0]
        print('<%s>: (tgt_only)' % tok, file=sys.stdout)
        print('CBoW  :', '\t'.join(tgt_only_outputs_cbow[tok].split('\t')[1:]))
        print('Linear:', '\t'.join(tgt_only_outputs_cross1[tok].split('\t')[1:]))
        print('LLM   :', '\t'.join(tgt_only_outputs_cross2[tok].split('\t')[1:]))
        print()
        # print(outputs_cross1[tok])
        # print(outputs_cross2[tok])
    print()

    # with open('shared.vocab', 'w') as f:
    #     for tok in shared_outputs_cross2:
    #         print(tok, file=f)

    # with open('tgt_only.vocab', 'w') as f:
    #     for tok in tgt_only_outputs_cross2:
    #         print(tok, file=f)
        
    sys.stdout = sys.__stdout__

    header = ['Model', 'shared', 'tgt-only', 'all']
    data = [['Linear'] + stat1, ['LLM'] + stat2]
    df = pd.DataFrame(data, columns=header).set_index('Model')
    print(df)

if __name__ == '__main__':
    cli()

    
