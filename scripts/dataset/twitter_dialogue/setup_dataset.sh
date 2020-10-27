#!/bin/bash

. ./const.sh
script_dir=$(cd $(dirname $0); pwd)
src_dir=$dataset_root/twitter/tokenized.mecab
shuffled_dir=$dataset_root/twitter/tokenized.mecab.shuf
shuffled_sp_dir=$dataset_root/twitter/tokenized.sp16000
data1=$shuffled_dir/2016-all # 17073331
data2=$shuffled_dir/2017-all # 17215519
suffixes=(dialogs tids uids utime idx)
data_types=(train dev test)

# Shuffle tokenized datasets.
if [ ! -e $shuffled_dir/2016-all.dialogs ]; then
    python scripts/dataset/twitter/random_shuffle.py \
	   2016-all \
	   -src $dataset_root/twitter/tokenized.mecab \
	   -tgt $shuffled_dir &
fi

if [ ! -e $shuffled_dir/2017-all.dialogs ]; then
    python scripts/dataset/twitter/random_shuffle.py \
	   2017-all \
	   -src $src_dir \
	   -tgt $shuffled_dir &
fi
wait

# Separate the datasets into train, dev, test sets.
if [ ! -e $shuffled_dir/train.dialogs ]; then
    for suffix in ${suffixes[@]}; do
	echo $suffix
	head -n 16973331 $data1.$suffix > $shuffled_dir/train.$suffix &
	tail -n 100000 $data1.$suffix > $shuffled_dir/dev.$suffix &
	head -n 100000 $data2.$suffix > $shuffled_dir/test.$suffix &
    done
fi

# Separate the dialogs into .src and .tgt files
for dtype in ${data_types[@]}; do
    if [ ! -e $shuffled_dir/$dtype.${num_turns}turn.src ]; then
	echo "Separating the concatenated dialogues into utterance and response parts."
	python scripts/dataset/twitter/separate_turns.py \
	       $shuffled_dir/$dtype.dialogs \
	       --num-turns $num_turns
    fi
done
if [ ! -e $shuffled_dir/train.${num_turns}turn.joined ]; then
    cp $shuffled_dir/train.${num_turns}turn.src \
       $shuffled_dir/train.${num_turns}turn.joined
    cat $shuffled_dir/train.${num_turns}turn.tgt \
	>> $shuffled_dir/train.${num_turns}turn.joined
fi
wait

for dtype in ${data_types[@]}; do
    for lang in (src tgt); do
	ln -sf $dtype.${num_turns}turn.$lang $shuffled_dir/$dtype.$lang
    done
done

./scripts/dataset/twitter/tokenize_sentencepiece.sh \
    $shuffled_dir \
    $shuffled_sp_dir \
    $n_sentencepiece \
    $num_turns




