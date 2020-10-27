#!/bin/bash

. ./const.sh
script_dir=$(cd $(dirname $0); pwd)
original_dir=$daily_data_root/ijcnlp_dailydialog
tokenized_dir=$daily_data_root/processed.moses
truecased_dir=$daily_data_root/processed.moses.truecased
num_turns=1

if [ ! -e $daily_data_root ]; then
    wget http://yanran.li/files/ijcnlp_dailydialog.zip -O $daily_data_root.zip
    unzip $daily_data_root.zip -d $daily_data_root
    rm $daily_data_root.zip 
    unzip $original_dir/train.zip -d $original_dir
    unzip $original_dir/validation.zip -d $original_dir
    unzip $original_dir/test.zip -d $original_dir
fi

if [ ! -e $original_dir/processed.${num_turns}turn ]; then
    python $script_dir/setup_dataset.py \
	   --num-turns $num_turns \
	   --source-dir $original_dir
fi

# Tokenization
if [ ! -e $tokenized_dir ]; then
    mkdir -p $tokenized_dir
fi

files=(train dev test)
suffixes=(src tgt)
for file in ${files[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $tokenized_dir/$file.$suffix ]; then
	    perl $tokenizer_path \
		 < $original_dir/processed.${num_turns}turn/$file.$suffix \
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
	    perl $truecaser_script_path \
		 --model $truecaser_model_path.en \
		 < $tokenized_dir/$file.$suffix \
		 > $truecased_dir/$file.$suffix &
	fi
    done
done

wait
cat $tokenized_dir/train.src > $truecased_dir/train.flat
cat $tokenized_dir/train.tgt >> $truecased_dir/train.flat

