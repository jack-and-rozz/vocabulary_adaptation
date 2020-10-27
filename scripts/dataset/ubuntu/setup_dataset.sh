#!/bin/bash

. ./const.sh
original_dir=$ubuntu_data_root/original
tokenized_dir=$ubuntu_data_root/processed.moses
truecased_dir=$ubuntu_data_root/processed.moses.truecased


if [ ! -e $ubuntu_data_root ]; then
    mkdir -p $ubuntu_data_root
fi
# if [ ! -e $ubuntu_data_root/ubuntu-ranking-dataset-creator ]; then
#     # This script requires python 2 with nltk, six, and unicodecsv.
#     cd $ubuntu_data_root
#     git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
#     ln -sf ubuntu-ranking-dataset-creator original
#     cd ubuntu-ranking-dataset-creator/src
#     ./generate.sh -t -s -l
# fi

# Separate multi-turn dialogs into single-turn dialogs. In this step, all portions are included as for the training set, and only the last turns are included as for the dev and test set. 
max_dialog_turns=1
script_dir=$(cd $(dirname $0); pwd)

if [ ! -e $original_dir/train.${max_dialog_turns}turn.src ]; then
    python $script_dir/setup_dataset.py -s $original_dir --num-turns ${max_dialog_turns} 
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
		 < $original_dir/$file.${max_dialog_turns}turn.$suffix \
		 | sed -e "s/_ _ eou _ _/__eou__/g" \
		 | sed -e "s/_ _ eot _ _/__eot__/g" \
		 | sed -e "s/&lt; URL &gt;/<URL>/g" \
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




exit 1
# if [ ! -e $processed_dir ]; then
#    mkdir -p $processed_dir
# fi

# # # Extract N-turn dialogs.
# # python scripts/ubuntu/setup_dataset.py -s $original_dir --max-turns ${max_dialog_turns}


# # Concat src and tgt.
# paste $original_dir/train.${max_dialog_turns}turns.src $original_dir/train.${max_dialog_turns}turns.tgt > $original_dir/train.${max_dialog_turns}turns
# paste $original_dir/dev.${max_dialog_turns}turns.src $original_dir/dev.${max_dialog_turns}turns.tgt > $original_dir/dev.${max_dialog_turns}turns
# paste $original_dir/test.${max_dialog_turns}turns.src $original_dir/test.${max_dialog_turns}turns.tgt > $original_dir/test.${max_dialog_turns}turns

# # Remove URLs.
# cat $original_dir/train.${max_dialog_turns}turns | grep -v '<URL>' > $original_dir/train.${max_dialog_turns}turns.nourl
# cat $original_dir/dev.${max_dialog_turns}turns | grep -v '<URL>' > $original_dir/dev.${max_dialog_turns}turns.nourl
# cat $original_dir/test.${max_dialog_turns}turns | grep -v '<URL>' > $original_dir/test.${max_dialog_turns}turns.nourl

# # Separate src and tgt.
# cut -f1 $original_dir/train.${max_dialog_turns}turns.nourl > $original_dir/train.${max_dialog_turns}turns.nourl.src
# cut -f2 $original_dir/train.${max_dialog_turns}turns.nourl > $original_dir/train.${max_dialog_turns}turns.nourl.tgt
# cut -f1 $original_dir/dev.${max_dialog_turns}turns.nourl > $original_dir/dev.${max_dialog_turns}turns.nourl.src
# cut -f2 $original_dir/dev.${max_dialog_turns}turns.nourl > $original_dir/dev.${max_d-e ialog_turns}turns.nourl.tgt
# cut -f1 $original_dir/test.${max_dialog_turns}turns.nourl > $original_dir/test.${max_dialog_turns}turns.nourl.src
# cut -f2 $original_dir/test.${max_dialog_turns}turns.nourl > $original_dir/test.${max_dialog_turns}turns.nourl.tgt

# # <dataset including URL>
# # Tokenize and fix special tokens.
# # perl $tokenizer_path < $original_dir/train.${max_dialog_turns}turns.src | sed -e "s/_ _ eou _ _/__eou__/g" | sed "s/_ _ eot _ _/__eot__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g" > $processed_dir/train.${max_dialog_turns}turns.src
# # perl $tokenizer_path < $original_dir/train.${max_dialog_turns}turns.tgt | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g" > $processed_dir/train.${max_dialog_turns}turns.tgt
# # perl $tokenizer_path < $original_dir/dev.${max_dialog_turns}turns.src   | sed -e "s/_ _ eou _ _/__eou__/g" | sed "s/_ _ eot _ _/__eot__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g" > $processed_dir/dev.${max_dialog_turns}turns.src
# # perl $tokenizer_path < $original_dir/dev.${max_dialog_turns}turns.tgt   | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g" > $processed_dir/dev.${max_dialog_turns}turns.tgt
# # perl $tokenizer_path < $original_dir/test.${max_dialog_turns}turns.src  | sed -e "s/_ _ eou _ _/__eou__/g" | sed "s/_ _ eot _ _/__eot__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g"> $processed_dir/test.${max_dialog_turns}turns.src
# # perl $tokenizer_path < $original_dir/test.${max_dialog_turns}turns.tgt  | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/\&lt; URL \&gt;/\&lt;URL\&gt;/g" > $processed_dir/test.${max_dialog_turns}turns.tgt

# # <dataset without URL>
# # Tokenize and fix special tokens.
# perl $tokenizer_path < $original_dir/train.${max_turns}turns.nourl.src | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/_ _ eot _ _/__eot__/g"  > $processed_dir/train.${max_turns}turns.src
# perl $tokenizer_path < $original_dir/train.${max_turns}turns.nourl.tgt | sed -e "s/_ _ eou _ _/__eou__/g" > $processed_dir/train.${max_turns}turns.tgt
# perl $tokenizer_path < $original_dir/dev.${max_turns}turns.nourl.src   | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/_ _ eot _ _/__eot__/g"  > $processed_dir/dev.${max_turns}turns.src
# perl $tokenizer_path < $original_dir/dev.${max_turns}turns.nourl.tgt   | sed -e "s/_ _ eou _ _/__eou__/g" > $processed_dir/dev.${max_turns}turns.tgt
# perl $tokenizer_path < $original_dir/test.${max_turns}turns.nourl.src  | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/_ _ eot _ _/__eot__/g" > $processed_dir/test.${max_turns}turns.src
# perl $tokenizer_path < $original_dir/test.${max_turns}turns.nourl.tgt  | sed -e "s/_ _ eou _ _/__eou__/g" > $processed_dir/test.${max_turns}turns.tgt

# cat $processed_dir/train.${max_turns}turns.src > $processed_dir/train.${max_turns}turns.all
# cat $processed_dir/train.${max_turns}turns.tgt >> $processed_dir/train.${max_turns}turns.all


