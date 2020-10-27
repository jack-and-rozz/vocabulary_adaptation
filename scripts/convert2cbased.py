import argparse


def main(args):
    delim = args.space_token
    src_filename = args.src_file.split('/')[-1]
    src = [' '.join(delim.join(l.strip().split())) for l in open(args.src_file)]
    tgt_filename = args.tgt_file.split('/')[-1]
    tgt = [' '.join(delim.join(l.strip().split())) for l in open(args.tgt_file)]

    with open(args.target_dir + '/' + src_filename, 'w') as f: 
        for l in src:
            f.write(l + '\n')

    with open(args.target_dir + '/' + tgt_filename, 'w') as f: 
        for l in tgt:
            f.write(l + '\n')

if __name__ == "__main__":
    desc = 'python scripts/convert2cbased.py dataset/aspec-je/original/train.en dataset/aspec-je/original/train.ja'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_file', type=str)
    parser.add_argument('tgt_file', type=str)
    parser.add_argument('target_dir', type=str)
    parser.add_argument('--space_token', default='▁', type=str)

    args = parser.parse_args()
    main(args)
