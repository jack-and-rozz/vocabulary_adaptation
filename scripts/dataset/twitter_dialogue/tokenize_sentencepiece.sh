#!/bin/bash

vocab_size=16000
source_dir=$1
target_dir=$2
vocab_size=$3
num_turns=$4
lang=ja
model=spm.$lang
data_types=(train dev test)
suffixes=(src tgt)
. ./const.sh

if [ ! -e $target_dir ]; then
    mkdir -p $target_dir
fi

if [ ! -e $target_dir/$model.model ]; then
    echo "training sentencepiece from $source_dir/$train_file..."
    spm_train --vocab_size $vocab_size \
	      --input $source_dir/train.${num_turns}turn.joined  \
	      --input_sentence_size 3000000 \
	      --shuffle_input_sentence true \
	      --model_prefix $target_dir/$model \
	      --pad_id 0 --unk_id 1 --bos_id 2 --eos_id 3 \
	      --unk_surface '▁<unk>' \
	      --user_defined_symbols '▁<EOT>' 
    ln -sf $model.model $target_dir/spm.src.model
    ln -sf $model.vocab $target_dir/spm.src.vocab
fi

for dtype in ${data_types[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $target_dir/$dtype.${num_turns}turn.$suffix ]; then
	    echo "encoding $source_dir/$dtype.${num_turns}turn.$suffix to pieces..."
	    spm_encode --model $target_dir/$model.model \
		       < $source_dir/$dtype.${num_turns}turn.$suffix \
		       > $target_dir/$dtype.${num_turns}turn.$suffix &
	fi
	ln -sf $dtype.${num_turns}turn.$suffix $target_dir/$dtype.$suffix
    done
done
wait


# Train embeddings.
emb_train_file=train.flat
emb_name=cbow.$lang.${emb_size}d
if [ ! -e $target_dir/train.${num_turns}turn.joined ]; then
    head -n 1500000 $target_dir/train.${num_turns}turn.src \
	 > $target_dir/$emb_train_file
    tail -n 1500000 $target_dir/train.${num_turns}turn.tgt \
	 >> $target_dir/$emb_train_file
fi

# if [ ! -e $target_dir/$emb_name ]; then
#     echo "training Word2vec from $target_dir/$emb_train_file..."
#     word2vec -size $emb_size -threads 24 \
# 	     -train $target_dir/$emb_train_file \
# 	     -output $target_dir/$emb_name \
# 	     -save-vocab $target_dir/$emb_name.vocab \
# 	     -min-count 1
# fi

