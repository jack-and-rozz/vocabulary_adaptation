#!/bin/bash

. ./const.sh
script_dir=$(cd $(dirname $0); pwd)
root_dir=$(pwd)

tgt_domain=common_ende
archive_filename=commoncrawl.de-en

tgt_data_root=$(eval echo '$'$tgt_domain'_data_root')
src_lang=$(eval echo '$'$tgt_domain'_src_lang')
tgt_lang=$(eval echo '$'$tgt_domain'_tgt_lang')

archive_dir=$tgt_data_root/archive
tokenized_dir=$tgt_data_root/processed.moses
truecased_dir=$tgt_data_root/processed.moses.truecased

if [ ! -e $archive_dir ]; then
    mkdir -p $archive_dir
fi

# TODO: download the dataset from wmt14
ln -sf $(pwd)/dataset/wmt14/archive/training/$archive_filename.$src_lang $archive_dir
ln -sf $(pwd)/dataset/wmt14/archive/training/$archive_filename.$tgt_lang $archive_dir

if [ ! -e $archive_dir/train.$src_lang ]; then
    head -n 2397123  $archive_dir/$archive_filename.$src_lang > $archive_dir/train.$src_lang &
    head -n 2397123  $archive_dir/$archive_filename.$tgt_lang > $archive_dir/train.$tgt_lang &
    tail -n 2000  $archive_dir/$archive_filename.$src_lang > $archive_dir/dev.$src_lang &
    tail -n 2000  $archive_dir/$archive_filename.$tgt_lang > $archive_dir/dev.$tgt_lang & 
    wait
fi
ln -sf dev.$src_lang $archive_dir/test.$src_lang
ln -sf dev.$tgt_lang $archive_dir/test.$tgt_lang


files=(train dev test)
suffixes=(en de)


# Tokenization
if [ ! -e $tokenized_dir ]; then
    mkdir -p $tokenized_dir
fi

for file in ${files[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $tokenized_dir/$file.$suffix ]; then
	    perl $root_dir/$tokenizer_path \
		 -l $suffix \
		 < $archive_dir/$file.$suffix \
		 > $tokenized_dir/$file.$suffix &

	fi
    done
done
wait 


# Truecasing
if [ ! -e $truecased_dir ]; then
    mkdir -p $truecased_dir
fi

for file in ${files[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $truecased_dir/$file.$suffix ]; then
	    perl $root_dir/$truecaser_script_path \
		 --model $root_dir/$truecaser_model_path.$suffix \
		 < $tokenized_dir/$file.$suffix \
		 > $truecased_dir/$file.$suffix &
	fi
    done
done
wait

ln -sf train.$src_lang $truecased_dir/monolingual.$src_lang
ln -sf train.$tgt_lang $truecased_dir/monolingual.$tgt_lang
