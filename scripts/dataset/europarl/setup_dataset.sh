#!/bin/bash

. ./const.sh
script_dir=$(cd $(dirname $0); pwd)
root_dir=$(pwd)
data_root_dir=$europarl_deen_data_root 
tgt_domain=europarl_ende
# archive_filename=commoncrawl.de-en

tgt_data_root=$(eval echo '$'$tgt_domain'_data_root')
src_lang=$(eval echo '$'$tgt_domain'_src_lang')
tgt_lang=$(eval echo '$'$tgt_domain'_tgt_lang')

archive_dir=$tgt_data_root/archive
tokenized_dir=$tgt_data_root/processed.moses
truecased_dir=$tgt_data_root/processed.moses.truecased

if [ ! -e $tgt_data_root ]; then
    mkdir -p $tgt_data_root
fi

if [ ! -e $archive_dir/de-en.tgz ]; then
    cd $archive_dir
    wget http://www.statmt.org/europarl/v7/de-en.tgz
    tar -zxvf de-en.tgz
    cd $root_dir
fi


if [ ! -e $archive_dir/common-test.tgz ]; then
    cd $archive_dir
    wget https://www.statmt.org/europarl/v1/common-test.tgz
    tar -zxvf common-test.tgz
    cd $root_dir
fi

cd $archive_dir/common-test
target_files=(ep-test-5-15.e ep-test-5-15.f.de)
for path in ${target_files[@]}; do
    if [[ $path =~ "utf8" ]]; then
	continue
    fi
    if [ ! -e $path.utf8 ]; then
	iconv -f iso-8859-1 -t utf8 $path > $path.utf8
    fi
done

cd $root_dir
cd $archive_dir
if [ ! -e train.en ]; then
    head -n 1918209 europarl-v7.de-en.en > train.en
    head -n 1918209 europarl-v7.de-en.de > train.de
fi
if [ ! -e dev.en ]; then
    tail -n 2000 europarl-v7.de-en.en > dev.en
    tail -n 2000 europarl-v7.de-en.de > dev.de
fi
if [ ! -e test.en ]; then
   cp common-test/ep-test-5-15.e.utf8 test.en
   cp common-test/ep-test-5-15.f.de.utf8 test.de
fi
cd $root_dir


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

