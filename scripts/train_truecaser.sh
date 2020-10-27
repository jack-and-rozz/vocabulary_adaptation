#!/bin/bash

# The news commentary corpus in WMT'13 is employed to train a domain-independent truecaser, following
# http://www.statmt.org/moses/manual/manual.pdf
# http://www.statmt.org/moses/?n=Moses.Baseline
. ./const.sh

if [ ! -e $moses_data_dir ]; then
    mkdir -p $moses_data_dir
fi

target_langs=(en fr de)
# Train English truecaser.
if [ ! -e $moses_data_dir/training/news-commentary-v8.fr-en.tok.en ]; then
    cd $moses_data_dir
    if [ ! -e training/news-commentary-v8.fr-en.en ]; then
	if [ ! -e training-parallel-nc-v8.tgz ]; then
	    wget http://www.statmt.org/wmt13/training-parallel-nc-v8.tgz
	fi
	tar zxvf training-parallel-nc-v8.tgz
    fi
    cd ../..
fi

other_langs=(fr de)
if [ ! -e $moses_data_dir/training/news-commentary-v8.fr-en.tok.en ]; then
    perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \
         < $moses_data_dir/training/news-commentary-v8.fr-en.en    \
	 > $moses_data_dir/training/news-commentary-v8.fr-en.tok.en
fi
for lang in ${other_langs[@]}; do
    if [ ! -e $moses_data_dir/training/news-commentary-v8.$lang-en.tok.$lang ]; then
	perl tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -l $lang \
             < $moses_data_dir/training/news-commentary-v8.$lang-en.$lang    \
	     > $moses_data_dir/training/news-commentary-v8.$lang-en.tok.$lang
    fi
done


if [ ! -e $moses_data_dir/truecase-model.en ]; then
   perl $moses_script_path/recaser/train-truecaser.perl \
	--model $moses_data_dir/truecase-model.en \
	--corpus $moses_data_dir/training/news-commentary-v8.fr-en.tok.en
fi
for lang in ${other_langs[@]}; do
    if [ ! -e $moses_data_dir/truecase-model.$lang ]; then
	perl $moses_script_path/recaser/train-truecaser.perl \
	     --model $moses_data_dir/truecase-model.$lang \
	     --corpus $moses_data_dir/training/news-commentary-v8.$lang-en.tok.$lang
    fi
done
