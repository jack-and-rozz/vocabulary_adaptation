
import argparse, random, sys
import numpy as np

import cupy
from cupy import cuda

def parse_embeddings(embed_path):
    embed_dict = {}
    with open(embed_path, errors='ignore') as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = np.array([float(weight) for weight in pieces[1:]])
    return embed_dict


def search_neighbors(src_batch_emb, tgt_all_emb,
                     word2id, batch_words,
                     include_self=False, num_neighbors=10):

    # Normalize batch embeddings
    src_batch_emb_norm = src_batch_emb / cupy.linalg.norm(src_batch_emb, axis=1)[:, None]

    # Compute cosine similarity
    cos_score = src_batch_emb_norm.dot(tgt_all_emb.T) # [batch_size, num_words]

    # Ignore exact matching words
    if not include_self:
        # indexはbatchの各単語のword indexをもっている
        word_index = cupy.array([word2id[word] for word in batch_words if word in word2id], dtype=cupy.int32)
        batch_index = cupy.array([i for i, word in enumerate(batch_words) if word in word2id], dtype=cupy.int32)
        # Set the score of matching words to very small
        cos_score[batch_index, word_index] = -100
    sim_indices = cupy.argsort(-cos_score, axis=1)[:, :num_neighbors] # [batch_size, num_neighbors]
    # conc = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.asnumpy(sim_cos_scores)
    return sim_indices, sim_cos_scores

def compute_similarity(src_batch_emb, tgt_all_emb,
                       word2id, batch_words, indices,
                       include_self=False, num_neighbors=10):

    # Normalize batch embeddings
    src_batch_emb_norm = src_batch_emb / cupy.linalg.norm(src_batch_emb, axis=1)[:, None]

    # Compute cosine similarity
    cos_score = src_batch_emb_norm.dot(tgt_all_emb.T) # [batch_size, num_words]

    sim_indices = indices
    # conc = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.asnumpy(sim_cos_scores)
    return sim_cos_scores

def main(args):
    batch_size = args.batch_size
    src_emb = parse_embeddings(args.src_emb)
    tgt_emb = parse_embeddings(args.tgt_emb)

    src_words = set(src_emb.keys())
    tgt_words = set(tgt_emb.keys())
    shared_words = list(src_words.intersection(tgt_words))
    shared_word2id = {k:i for i, k in enumerate(shared_words)}

    src_emb_shared = np.array([src_emb[k] for k in shared_words], 
                              dtype=np.float32)
    tgt_emb_shared = np.array([tgt_emb[k] for k in shared_words],
                              dtype=np.float32)

    src_emb_shared_normed = cupy.array(src_emb_shared / np.linalg.norm(src_emb_shared, axis=1)[:, None])
    tgt_emb_shared_normed = cupy.array(tgt_emb_shared / np.linalg.norm(tgt_emb_shared, axis=1)[:, None])
    src_batched_emb = cupy.empty((batch_size, src_emb_shared.shape[1]), 
                                 dtype=cupy.float32)

    sim_differences = np.empty((len(shared_words)), dtype=np.float32)

    for i in range(0, len(shared_words), batch_size):
        s = src_emb_shared[i:i+batch_size]
        src_batched_words = shared_words[i:i+batch_size]
        src_batched_emb[:len(s), :] = cupy.array(s)

        # Compute neighbors and their cosine similarity in the source space
        src_neighbors, src_cos_scores = search_neighbors(
            src_batched_emb, src_emb_shared_normed, 
            shared_word2id, src_batched_words,
            num_neighbors=args.num_neighbors)

        # Compute the neighbors' cosine similarity in the target space
        tgt_cos_scores = compute_similarity(
            src_batched_emb, tgt_emb_shared_normed, 
            shared_word2id, src_batched_words, src_neighbors,
            num_neighbors=args.num_neighbors)

        # Compute the difference of similarlity
        sim_diff_mean = np.mean(np.abs(src_cos_scores - tgt_cos_scores), axis=-1)
        sim_differences[i:i+len(s)] = sim_diff_mean[:len(s)]

    sim_indice = np.argsort(sim_differences)

    for idx in sim_indice[:int(args.keep_prob * len(sim_differences))]:
        print(shared_words[idx], shared_words[idx])

    # for idx in sim_indice:
    #     print(shared_words[idx], sim_differences[idx])

    # print('# src, tgt, shared words =', len(src_words), len(tgt_words), len(shared_words))

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_emb', type=str)
    parser.add_argument('tgt_emb', type=str)
    
    parser.add_argument('--num_neighbors', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--keep_prob', default=0.5, type=float,
                        help='Percentage to keep shared tokens of which neighbors do not change across two spaces')
    args = parser.parse_args()
    main(args)
