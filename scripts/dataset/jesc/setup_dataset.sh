#!/bin/bash

. ./const.sh

root_dir=$(pwd)
script_dir=$(cd $(dirname $0); pwd)
jesc_data_root=dataset/jesc-je
original_dir=$jesc_data_root/original
tokenized_dir=$jesc_data_root/processed.kytea-moses
truecased_dir=$jesc_data_root/processed.kytea-moses.truecased

# You need to download split.tar.gz from https://nlp.stanford.edu/projects/jesc/data/split.tar.gz and decompress it to $original_dir.

dtypes=(train dev test)


# Split the original archives into train.en, train.ja, dev.en, ... 
if [ ! -e $original_dir ]; then
    mkdir -p $original_dir
fi

for dtype in ${dtypes[@]}; do
    if [ ! -e $original_dir/$dtype.en ] || [ ! -e $original_dir/$dtype.ja ]; then
	if [ ! -e $original_dir/split ]; then
	    if [ -e $original_dir/split.tar.gz ]; then
		cd $original_dir
		tar -zxvf split.tar.gz
		cd $root_dir
	    else
		echo "Error: $original_dir/split was not found. This script requires concatenated parallel corpora ($original_dir/train,  $original_dir/dev, and $original_dir/test)." 
		echo "Download split.tar.gz from https://nlp.stanford.edu/projects/jesc/data/split.tar.gz and decompress it to $jesc_data_root."
		exit 1
	    fi
	fi
	cut -f1 $original_dir/split/$dtype > $original_dir/$dtype.en &
	cut -f2 $original_dir/split/$dtype > $original_dir/$dtype.ja &
    fi
done
wait 

# Tokenization
if [ ! -e $tokenized_dir ]; then
    mkdir -p $tokenized_dir
fi

if [ ! -e $tokenizer_path ]; then
    echo "Error: $tokenizer_path was not found. Download Moses Toolkit from https://github.com/moses-smt/mosesdecoder."
    exit 1

fi
for dtype in ${dtypes[@]}; do
    if [ ! -e $tokenized_dir/$dtype.en ]; then
	 perl $tokenizer_path -l en \
	      < $original_dir/$dtype.en \
	      > $tokenized_dir/$dtype.en &
    fi

    if [ ! -e $tokenized_dir/$dtype.ja ]; then
	kytea -notags \
	      < $original_dir/$dtype.ja \
	      > $tokenized_dir/$dtype.ja &
    fi
done
wait 


# Truecasing
if [ ! -e $truecased_dir ]; then
    mkdir -p $truecased_dir
fi

if [ ! -e $truecaser_script_path ]; then
    echo "Error: $truecaser_script_path was not found. Download Moses Toolkit from https://github.com/moses-smt/mosesdecoder."
    exit 1
fi 

for dtype in ${dtypes[@]}; do
    if [ ! -e $truecased_dir/$dtype.en ]; then
	if [ ! -e $truecaser_model_path.en ]; then
	    echo "Error: $truecaser_model_path.en was not found. Train a model for truecasing following the official tutorial (http://www.statmt.org/moses/?n=Moses.Baseline)."
	    exit 1
	fi
	perl $truecaser_script_path \
	     --model $truecaser_model_path.en \
	     < $tokenized_dir/$dtype.en \
	     > $truecased_dir/$dtype.en &
    fi
    if [ ! -e $truecased_dir/$dtype.ja ]; then
	python -c "import unicodedata; [print(unicodedata.normalize('NFKC', s).strip()) for s in open('$tokenized_dir/$dtype.ja')]" > $truecased_dir/$dtype.ja &
    fi
done
wait

ln -sf train.ja $truecased_dir/train.all.ja
ln -sf train.en $truecased_dir/train.all.en
wait


# Subword tokenization was moved to another script.



