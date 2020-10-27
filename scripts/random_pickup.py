import argparse, random, sys

def main(args):
    random.seed(args.seed)
    src = [l for l in open(args.src_file)]
    tgt = [l for l in open(args.tgt_file)]
    output_header = '.'.join(args.src_file.split('.')[:-1])
    src_suffix = '.' + args.src_file.split('.')[-1]
    tgt_suffix = '.' + args.tgt_file.split('.')[-1]


    if args.N[-1].lower() == 'k':
        N = int(args.N[:-1]) * 1000
    else:
        N = int(args.N)

    if N < 1000:
        raise ValueError('N cannot be less than 1000.')
    else:
        size_suffix = '.' + str(int(N / 1000)) + 'k'
    assert len(src) == len(tgt)

    n_samples = len(src)
    
    if N > n_samples:
        print('args.N must be smaller than or equal to the size of the original file.', file=sys.stderr)
        exit(1)

    picked_indices = random.sample(range(n_samples), N)

    with open(output_header + size_suffix + src_suffix, 'w') as f:
        for idx in picked_indices:
            f.write(src[idx])

    with open(output_header + size_suffix + tgt_suffix, 'w') as f:
        for idx in picked_indices:
            f.write(tgt[idx])

    with open(output_header + size_suffix + '.idx', 'w') as f:
        for idx in picked_indices:
            f.write(str(idx) + '\n')


if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_file', type=str)
    parser.add_argument('tgt_file', type=str)
    parser.add_argument('N', type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
 
# python scripts/random_pickup.py dataset/ubuntu-dialog/processed.moses.nourl/train.3turns.src dataset/ubuntu-dialog/processed.moses.nourl/train.3turns.tgt 100000
