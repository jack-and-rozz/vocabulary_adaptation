import argparse, random, sys, os
from orderedset import OrderedSet as ods

def main(args):
    random.seed(args.seed)
    src_lang_data = [x for x in open(args.src_lang_file)]
    tgt_lang_data = [x for x in open(args.tgt_lang_file)]
    src_lang = args.src_lang_file.split('.')[-1]
    tgt_lang = args.tgt_lang_file.split('.')[-1]

    assert len(src_lang_data) == len(tgt_lang_data)

    ft_indice = ods([int(x) for x in open(args.indice_of_ft_data)])
    mono_indice = ods(range(len(src_lang_data))) - ft_indice

    src_lang_indice = ods(random.sample(mono_indice, int(len(mono_indice)/2)))
    tgt_lang_indice = mono_indice - src_lang_indice

    assert len(src_lang_indice.intersection(ft_indice)) == 0 
    assert len(tgt_lang_indice.intersection(ft_indice)) == 0 

    if args.overwrite or (not os.path.exists(args.output_header + '+ft.' + src_lang)):
        with open(args.output_header + '+ft.' + src_lang, 'w') as f:
            for idx in src_lang_indice:
                f.write(src_lang_data[idx])
            for idx in ft_indice:
                f.write(src_lang_data[idx])

    if args.overwrite or (not os.path.exists(args.output_header + '+ft.' + tgt_lang)):
        with open(args.output_header + '+ft.' + tgt_lang, 'w') as f:
            for idx in tgt_lang_indice:
                f.write(tgt_lang_data[idx])
            for idx in ft_indice:
                f.write(tgt_lang_data[idx])

    if args.overwrite or (not os.path.exists(args.output_header + '.' + src_lang)):
        with open(args.output_header + '.' + src_lang, 'w') as f:
            for idx in src_lang_indice:
                f.write(src_lang_data[idx])

    if args.overwrite or (not os.path.exists(args.output_header + '.' + tgt_lang)):
        with open(args.output_header + '.' + tgt_lang, 'w') as f:
            for idx in tgt_lang_indice:
                f.write(tgt_lang_data[idx])


    if args.overwrite or (not os.path.exists(args.output_header + '.' + src_lang + '.idx')):
        with open(args.output_header + '.' + src_lang + '.idx', 'w') as f:
            for idx in src_lang_indice:
                f.write('%d\n' % idx)


    if args.overwrite or (not os.path.exists(args.output_header + '.' + tgt_lang + '.idx')):
        with open(args.output_header + '.' + tgt_lang + '.idx', 'w') as f:
            for idx in tgt_lang_indice:
                f.write('%d\n' % idx)




if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('src_lang_file', type=str)
    parser.add_argument('tgt_lang_file', type=str)
    parser.add_argument('indice_of_ft_data', type=str)
    parser.add_argument('output_header', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
