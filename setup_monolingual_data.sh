#!/bin/bash
echo "Running '$0 $@'..."
usage() {
    echo "Usage:$0 mode task"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

. ./const.sh $mode $task

mode=$1
task=$2

is_valid=$(validate_mode $mode $task)

root_dir=$(pwd)
size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_domain_wd=$(remove_tok_suffix $src_domain)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)
src_data_dir_wd=$(get_data_dir $mode $src_domain_wd)
tgt_data_dir_wd=$(get_data_dir $mode $tgt_domain_wd)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 



type2(){
    # Use target-domain fine-tuning set + half of the target-domain training data
    if [ ! -e $tgt_data_dir_wd/train.$size.$src_lang ]; then
	python scripts/random_pickup.py \
	       $tgt_data_dir_wd/train.${src_lang} \
	       $tgt_data_dir_wd/train.${tgt_lang} \
	       $size --seed $random_seed
    fi
    if [ ! -e $tgt_data_dir_wd/mono.ft$size.$src_lang ]; then
	python scripts/create_simulated_monolingual.py \
	       $tgt_data_dir_wd/train.${src_lang} \
	       $tgt_data_dir_wd/train.${tgt_lang} \
	       $tgt_data_dir_wd/train.$size.idx \
	       $tgt_data_dir_wd/mono.ft$size \
	       --seed $random_seed
    fi
}

type2
