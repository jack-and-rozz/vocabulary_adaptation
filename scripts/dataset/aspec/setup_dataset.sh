#!/bin/bash

# TODO: truecasingまで


. ./const.sh

script_dir=$(cd $(dirname $0); pwd)
aspec_data_root=$dataset_root/aspec-je
original_dir=$aspec_data_root/original
tokenized_dir=$aspec_data_root/processed.kytea-moses
truecased_dir=$aspec_data_root/processed.kytea-moses.truecased
dtypes=(train dev test)


if [ ! -e $original_dir/train-1.ja ] || [ ! -e $original_dir/train-2.ja ] || [ ! -e $original_dir/dev.ja ] || [ ! -e $original_dir/test.ja ]; then
    echo "$original_dir/*.ja or $original_dir/*.en were not found."
    echo "Download data from http://lotus.kuee.kyoto-u.ac.jp/ASPEC/ and move the train-{1,2,3}/dev/test files to $original_dir."
    exit 1
fi


# In experiments, only the first and the second portions of the training data were used due to the low quality of the third portion.
if [ ! -e $original_dir/train.en ]; then
    cat $original_dir/train-1.en > $original_dir/train.en
    cat $original_dir/train-2.en >> $original_dir/train.en 
fi
if [ ! -e $original_dir/train.ja ]; then
    cat $original_dir/train-1.ja > $original_dir/train.ja
    cat $original_dir/train-2.ja >> $original_dir/train.ja 
fi

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

ln -sf train.ja $truecased_dir/train.all.ja
ln -sf train.en $truecased_dir/train.all.en
wait








