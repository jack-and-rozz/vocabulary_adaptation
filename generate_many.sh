#!/bin/bash

usage() {
    echo "Usage:$0 src_domain tgt_domain task"
    exit 1
}

if [ $# -lt 3 ];then
    usage;
fi

. ./const.sh

src_domain=$1
tgt_domain=$2
task=$3

# finetune_sizes=(10k 100k 1000k all)
# finetune_sizes=(10k 100k all)
finetune_sizes=(10k)

src_vocab_size=16000
tgt_vocab_size=16000
if [[ $src_domain =~ ${sp_suffix} ]]; then
    direction=${src_domain}${src_vocab_size}${direction_tok}${tgt_domain}${tgt_vocab_size}
else
    direction=${src_domain}${sp_suffix}${src_vocab_size}${direction_tok}${sp_suffix}${tgt_domain}${tgt_vocab_size}.
fi


# Out-domain
if [ $src_domain != $tgt_domain ]; then
    ./setup_sentencepiece.sh $direction.noadapt.all $task
    ./preprocess.sh $direction.noadapt.all $task
    ./generate.sh $direction.noadapt.all $task
fi
for mode in $(ls $ckpt_root | grep $tgt_domain); do
    echo $mode
    if [[ $mode =~ .bak ]]; then
	continue
    fi
    # if [[ $mode =~ ${outdomain_ext} ]]; then
    # 	continue
    # fi
    if [[ $mode =~ ${backtranslation_ext}_aug ]]; then
	continue
    fi

    ./preprocess.sh $mode $task

    echo "./generate.sh $mode $task"
    ./generate.sh $mode $task
done
