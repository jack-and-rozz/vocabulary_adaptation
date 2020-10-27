#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

. ./const.sh
mode=$1
task=$2
model_root=$ckpt_root/$mode
if [ ! -e $model_root ]; then
    echo "$model_root does not exist."
    exit 1
fi

src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

data_dir=$(get_data_dir $mode $tgt_domain)
srcdict=$data_dir/fairseq.all/dict.$src_lang.txt

if [ $task == translation ]; then
    tgtdict=$data_dir/fairseq.all/dict.$tgt_lang.txt
fi

if [ ! -e $model_root/embeddings ];then
    mkdir $model_root/embeddings
fi
if [ -e $model_root/embeddings/encoder.indomain ] && [ -e $model_root/embeddings/decoder.indomain ]; then
    exit 1
fi

if [ ! -e $data_dir/fairseq.all ]; then
    ./preprocess.sh $model_root $task
fi

echo "Output trained model's embeddings to '$model_root/embeddings'."
if [ -z $tgtdict ];then
    python fairseq/load_trained_embeddings.py $model_root \
	   $srcdict \
	   --user-dir ${fairseq_user_dir} 

else
    python fairseq/load_trained_embeddings.py $model_root \
       $srcdict \
       --tgtdict $tgtdict \
       --user-dir ${fairseq_user_dir} 

fi

