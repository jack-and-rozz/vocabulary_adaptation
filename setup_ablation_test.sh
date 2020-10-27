#!/bin/bash
# This script trains and applies subword tokenization for the source (and the target) domain(s), respectively. As for the datasets which are constructed from the combination of multiple datasets, such as ones for multi-domain learning or back-translation, subword tokenization should be done in ./setup_*** by using subword-level datasets created by this script. 

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
if [ -n "$is_valid" ]; then
    echo $is_valid
    exit 1
fi

root_dir=$(pwd)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)

src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)
tgt_data_dir_wd=$(get_data_dir $mode $tgt_domain_wd)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

size=$(parse_size $mode)
files=(train.$size dev test)

if [[ $mode =~ \.${vocabadapt_ext}_enc\. ]]; then 
    ab_tgt_data_dir=$tgt_data_dir.enc
    adapt_lang=$src_lang
    unadapt_lang=$tgt_lang
elif [[ $mode =~ \.${vocabadapt_ext}_dec\. ]]; then 
    ab_tgt_data_dir=$tgt_data_dir.dec
    adapt_lang=$tgt_lang
    unadapt_lang=$src_lang
else
    echo "invalid mode: $mode"
    exit 1 
fi

if [ ! -e $ab_tgt_data_dir ]; then
    mkdir -p $ab_tgt_data_dir
fi


ln -sf $root_dir/$tgt_data_dir/spm.$adapt_lang.model $ab_tgt_data_dir
ln -sf $root_dir/$tgt_data_dir/spm.$adapt_lang.vocab $ab_tgt_data_dir
ln -sf $root_dir/$tgt_data_dir/word2vec.$adapt_lang.${emb_size}d $ab_tgt_data_dir
ln -sf $root_dir/$tgt_data_dir/word2vec.$adapt_lang.${emb_size}d.vocab $ab_tgt_data_dir
ln -sf $root_dir/$tgt_data_dir/dict.$adapt_lang.txt $ab_tgt_data_dir

ln -sf $root_dir/$src_data_dir/spm.$unadapt_lang.model $ab_tgt_data_dir
ln -sf $root_dir/$src_data_dir/spm.$unadapt_lang.vocab $ab_tgt_data_dir
ln -sf $root_dir/$src_data_dir/word2vec.$unadapt_lang.${emb_size}d $ab_tgt_data_dir
ln -sf $root_dir/$src_data_dir/word2vec.$unadapt_lang.${emb_size}d.vocab $ab_tgt_data_dir
ln -sf $root_dir/$src_data_dir/dict.$unadapt_lang.txt $ab_tgt_data_dir

for file in ${files[@]}; do
    ln -sf $root_dir/$tgt_data_dir/$file.$adapt_lang $ab_tgt_data_dir
    if [ ! -e $ab_tgt_data_dir/$file.$unadapt_lang ]; then
	spm_encode --model $ab_tgt_data_dir/spm.$unadapt_lang.model \
		   --output $ab_tgt_data_dir/$file.$unadapt_lang \
		   < $tgt_data_dir_wd/$file.$unadapt_lang &
    fi
done
wait
ln -sf $root_dir/$tgt_data_dir/monolingual.$adapt_lang $ab_tgt_data_dir

if [ ! -e $ab_tgt_data_dir/monolingual.$unadapt_lang ]; then 
    spm_decode --model $tgt_data_dir/spm.$unadapt_lang.model \
	       --output $ab_tgt_data_dir/monolingual.$unadapt_lang.tmp \
	       < $tgt_data_dir/monolingual.$unadapt_lang 
    spm_encode --model $ab_tgt_data_dir/spm.$unadapt_lang.model \
	       --output $ab_tgt_data_dir/monolingual.$unadapt_lang \
	       < $ab_tgt_data_dir/monolingual.$unadapt_lang.tmp
    rm $ab_tgt_data_dir/monolingual.$unadapt_lang.tmp
fi
