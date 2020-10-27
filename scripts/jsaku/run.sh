#!/bin/bash

usage() {
    echo "Usage:$0 [100k|100kmono]"
    exit 1
}
if [ $# -lt 1 ];then
    usage;
fi

tgt_data_size=$1
root_dir=../
format=plain
num_nn=5

# lang=en
# side=encoder
lang=ja
side=decoder

nmt_vec=${side}.indomain
linear_vec=${side}.aspec_sp16000_${tgt_data_size}@jesc_sp16000.linear-idt
llm_vec=${side}.aspec_sp16000_${tgt_data_size}@jesc_sp16000.llm-idt.nn10


src_nmt_dir=$root_dir/checkpoints/20200904.camera_ready/jesc_sp16000.outD.all/embeddings
tgt_data_dir=$root_dir/dataset/aspec-je/processed.kytea-moses.truecased/sp/v_aspec_sp16000_$tgt_data_size

ln -sf $src_nmt_dir src-nmt
src_nmt_dir=src-nmt # 何故か分からないがリンクを貼らないとembeddings.pyがエラーを吐く
results_dir=neighbors.$tgt_data_size.$lang

if [ ! -e $results_dir ]; then
    mkdir $results_dir
fi


if [ ! -e $results_dir/neighbor_candidates.$lang ]; then
    python embeddings.py listup-neighbor-candidates \
	   $tgt_data_dir/word2vec.$lang.512d \
	   $src_nmt_dir/$nmt_vec \
	   > $results_dir/neighbor_candidates.$lang
fi

if [ ! -e $results_dir/tgt-cbow.$lang ]; then
    python embeddings.py nn \
    	   $tgt_data_dir/word2vec.$lang.512d \
    	   $tgt_data_dir/word2vec.$lang.512d \
    	   --candidates-vocab-path $results_dir/neighbor_candidates.$lang \
    	   --format $format \
    	   --k $num_nn \
    	   > $results_dir/tgt-cbow.$lang &
fi
if [ ! -e $results_dir/src-nmt.$lang ]; then
    python embeddings.py nn \
    	   $src_nmt_dir/$nmt_vec \
    	   $src_nmt_dir/$nmt_vec \
    	   --candidates-vocab-path $results_dir/neighbor_candidates.$lang \
    	   --format $format \
    	   --k $num_nn \
    	   > $results_dir/src-nmt.$lang &
fi
if [ ! -e $results_dir/linear-idt.$lang ]; then
    python embeddings.py nn \
    	   $src_nmt_dir/$linear_vec \
    	   $src_nmt_dir/$nmt_vec \
 	   --candidates-vocab-path $results_dir/neighbor_candidates.$lang \
	   --format $format \
	   --k $num_nn \
    	   > $results_dir/linear-idt.$lang & 
fi
if [ ! -e $results_dir/llm-idt.$lang ]; then
    python embeddings.py nn \
    	   $src_nmt_dir/$llm_vec \
    	   $src_nmt_dir/$nmt_vec \
 	   --candidates-vocab-path $results_dir/neighbor_candidates.$lang \
	   --format $format \
	   --k $num_nn \
    	   > $results_dir/llm-idt.$lang &
fi
wait
python embeddings.py nn-overlaps \
       $results_dir/tgt-cbow.$lang \
       $results_dir/linear-idt.$lang \
       $results_dir/llm-idt.$lang \
       $results_dir/src-nmt.$lang \
       --format $format \
       --k $num_nn \
       > $results_dir/shared.summary \
       2> $results_dir/tgt_only.summary


