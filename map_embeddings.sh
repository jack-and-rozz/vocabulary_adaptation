#!/bin/bash
usage() {
    echo "Usage:$0 mode task"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
    exit 1
fi

. ./const.sh $mode $task
mode=$1
task=$2

src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
emb_type=$(parse_emb_type $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 
fixed=$(parse_fixed $mode)

src_spm_domain=$(parse_spm_domain $mode src)
src_spm_mono_size=$(parse_spm_mono_size $mode src)
tgt_spm_domain=$(parse_spm_domain $mode tgt)
tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

llm_nn=$(parse_llm_nn $mode)
if [ -z $llm_nn ]; then
    llm_nn=$llm_nn_default
fi

direction=${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}@${src_domain}${src_vocab_size}

###########################################
###             Setup
###########################################

echo "Running '$0 $1 $2 $3'..."
# Load the model's embeddings trained in the source domain.

if [[ $src_domain =~ _${sp_suffix} ]]; then
    src_model=${src_domain}${src_vocab_size}.${outdomain_ext}.all
else
    src_model=${src_domain}.${outdomain_ext}.all
fi

if [ ! -z $fixed ]; then
    # When this option is enabled, map the target domain embeddings to the source domain model of which embedding layer is fixed to the CBoW vectors.
    src_model=$src_model.fixed
fi

./load_trained_embeddings.sh $src_model $task
model_emb_dir=$ckpt_root/$src_model/embeddings

src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)

enc_src_emb=$model_emb_dir/encoder.indomain # the model trained in the src domain.
enc_tgt_emb=$tgt_data_dir/word2vec.$src_lang.512d # the embeddings trained by a raw corpus of the tgt domain.
enc_mapped_emb=$model_emb_dir/encoder.$direction.$emb_type

if [[ $emb_type =~ llm ]]; then
    enc_mapped_emb=$enc_mapped_emb.nn${llm_nn}
fi

if [ ! -e $enc_tgt_emb ]; then
    echo "$enc_tgt_emb was not found. Run './train_cbow.sh ...' with the raw corpus of the tgt domain."
    exit 1 
fi


if [ $task == translation ];then
    dec_src_emb=$model_emb_dir/decoder.indomain
    dec_tgt_emb=$tgt_data_dir/word2vec.$tgt_lang.512d
    dec_mapped_emb=$model_emb_dir/decoder.$direction.$emb_type
    if [[ $emb_type =~ llm ]]; then
	dec_mapped_emb=$dec_mapped_emb.nn${llm_nn}
    fi
fi


###########################################
###    Mapping Encoder's Embeddings
###########################################

N_SPECIAL_WORDS=4 # Special words(<pad>, </s>, <unk>, '<s>')

# *Note*
# In the following, the 'src' or 'tgt' domains are reversely considered. 
# We map tgt-CBoW embeddings to src-NMT embeddings.

dict_keep_prob=0.5

echo "Mapping encoder's embeddings from $enc_tgt_emb to $enc_src_emb. The mapped embeddings are saved as $enc_mapped_emb."
case $emb_type in
    nomap)
	cp $enc_tgt_emb $enc_mapped_emb.tmp
	;;
    llm-dict*)
	# Use only domain-invariant words for mapping.
	enc_dict=$model_emb_dir/encoder.kp$dict_keep_prob.dict
	if [ ! -e $enc_dict ]; then
	    python scripts/pickup_domain_invariant_words.py \
		   $enc_tgt_emb $enc_src_emb \
		   --keep_prob $dict_keep_prob \
		   > $enc_dict
	fi
	if [ ! -e $enc_mapped_emb ]; then
	   python tools/llm/llm.py \
	   	  $enc_tgt_emb $enc_src_emb \
	   	  --ignore-exact-words \
	   	  --num-neighbors $llm_nn \
		  --map-dict-path $enc_dict \
	   	  < $enc_tgt_emb \
	   	  > $enc_mapped_emb.tmp
	fi
	;;
    # linear-dict*)
    # 	# Use only domain-invariant words for mapping.
    # 	enc_dict=$model_emb_dir/encoder.kp$dict_keep_prob.dict
    # 	if [ ! -e $enc_dict ]; then
    # 	    python scripts/pickup_domain_invariant_words.py \
    # 		   $enc_tgt_emb $enc_src_emb \
    # 		   --keep_prob $dict_keep_prob \
    # 		   > $enc_dict
    # 	fi
    # 	exit 1
    # 	if [ ! -e $enc_mapped_emb ]; then
    # 	    python tools/vecmap/map_embeddings.py \
    # 		   --init_identical --orthogonal --cuda \
    # 		   $enc_tgt_emb $enc_src_emb \
    # 		   $enc_mapped_emb.tmp $enc_tgt_emb.$direction.$emb_type
    # 	    rm $enc_tgt_emb.$direction.$emb_type # not used for our experiments.
    # 	fi
    # 	# TODO
    # 	;;

    llm-idt*)
	if [ ! -e $enc_mapped_emb ]; then
	   python tools/llm/llm.py \
	   	  $enc_tgt_emb $enc_src_emb \
	   	  --ignore-exact-words \
	   	  --num-neighbors $llm_nn \
	   	  < $enc_tgt_emb \
	   	  > $enc_mapped_emb.tmp
	fi
	;;
    llm-idt-v2)
	if [ ! -e $enc_mapped_emb ]; then
	    enc_src_emb_aligned=$enc_src_emb.${src_domain}2${tgt_domain}.aligned
	    enc_tgt_emb_aligned=$enc_tgt_emb.${src_domain}2${tgt_domain}.aligned

	    script_dir=tools/llm-v2
	    python $script_dir/scripts/align.py \
		--src-input-path $enc_tgt_emb \
		--trg-input-path $enc_src_emb \
		--src-aligned-path $enc_tgt_emb_aligned \
		--trg-aligned-path $enc_src_emb_aligned 

	    llm_map \
		   $enc_tgt_emb_aligned $enc_src_emb_aligned \
		   --config-path $script_dir/configs/bispace.base.json \
		   --config-log $script_dir/configs/$direction.json \
		   --detail-path $script_dir/$direction.npz \
		   -o reconstructor.type=\"bispace\" \
		   -o mapper.type=\"bispace\" \
		   -o nn_searcher.src.type=\"csls-topk\"  \
		   -o nn_searcher.input.type=\"csls-topk\"  \
		   -o nn_searcher.src.csls=$llm_nn \
		   -o nn_searcher.input.csls=$llm_nn \
		   -o nn_searcher.src.topk=$llm_nn \
		   -o nn_searcher.input.topk=$llm_nn \
		   < $enc_tgt_emb > $enc_mapped_emb.tmp
	    rm $enc_src_emb_aligned
	    rm $enc_tgt_emb_aligned
	fi
	;;
    linear-idt)
	if [ ! -e $enc_mapped_emb ]; then
	    python tools/vecmap/map_embeddings.py \
		   --init_identical --orthogonal --cuda \
		   $enc_tgt_emb $enc_src_emb \
		   $enc_mapped_emb.tmp $enc_tgt_emb.$direction.linear-idt
	    rm $enc_tgt_emb.$direction.linear-idt # not used for our experiments.
	fi
	;;
    * ) echo "invalid emb_type: $emb_type"
        exit 1 ;;
esac

if [ ! -e $enc_mapped_emb ] && [ -e $enc_mapped_emb.tmp ]; then

    # The embeddings of the special words are kept even after mapping.
    # 1. Copy the special word embeddings from the target (model's embs)
    sed -n 2,$(($N_SPECIAL_WORDS+1))p $enc_src_emb  > $enc_mapped_emb 

    # 2. Gather the tmpfile of mapped embeddings if the word is not a special word.
    sed -n 2,$(wc -l $enc_mapped_emb.tmp | cut -f1 -d ' ')p $enc_mapped_emb.tmp | grep -v '<pad>' | grep -v '</s>' | grep -v '<unk>'  | grep -v '<s>'   >> $enc_mapped_emb 
    # 3. Count the number of total words again, and insert it to the head.
    n_vocab=$(wc -l $enc_mapped_emb | cut -f1 -d ' ')
    n_dims=$(head -n1 $enc_mapped_emb.tmp | cut -f2 -d ' ') 
    sed -i "1i $n_vocab $n_dims" $enc_mapped_emb
    rm $enc_mapped_emb.tmp
fi

# Finish if the source and target languages are same.
if [ $task != translation ];then
    exit 1
fi


###########################################
###    Mapping Decoder's Embeddings
###########################################

echo "Mapping decoder's embeddings from $dec_tgt_emb to $dec_src_emb. The mapped embeddings are saved as $dec_mapped_emb."
case $emb_type in
    nomap)
	if [ ! -e $dec_mapped_emb ]; then
	    cp $dec_tgt_emb $dec_mapped_emb.tmp
	fi
	;;
    llm-dict*)
	# Use only domain-invariant words for mapping.
	dec_dict=$model_emb_dir/decoder.kp$dict_keep_prob.dict
	if [ ! -e $dec_dict ]; then
	    python scripts/pickup_domain_invariant_words.py \
		   $dec_tgt_emb $dec_src_emb \
		   --keep_prob $dict_keep_prob \
		   > $dec_dict
	fi
	if [ ! -e $dec_mapped_emb ]; then
	   python tools/llm/llm.py \
	   	  $dec_tgt_emb $dec_src_emb \
	   	  --ignore-exact-words \
	   	  --num-neighbors $llm_nn \
		  --map-dict-path $dec_dict \
	   	  < $dec_tgt_emb \
	   	  > $dec_mapped_emb.tmp
	fi
	;;
    linear-dict*)
	# Use only domain-invariant words for mapping.
	dec_dict=$model_emb_dir/decoder.kp$dict_keep_prob.dict
	if [ ! -e $dec_dict ]; then
	    python scripts/pickup_domain_invariant_words.py \
		   $dec_tgt_emb $dec_src_emb \
		   --keep_prob $dict_keep_prob \
		   > $dec_dict
	fi
	exit 1
	if [ ! -e $dec_mapped_emb ]; then
	    python tools/vecmap/map_embeddings.py \
		   --init_identical --orthogonal --cuda \
		   $dec_tgt_emb $dec_src_emb \
		   $dec_mapped_emb.tmp $dec_tgt_emb.$direction.$emb_type
	    rm $dec_tgt_emb.$direction.$emb_type # not used for our experiments.
	fi
	# TODO
	;;

    llm-idt)
	if [ ! -e $dec_mapped_emb ]; then
	   python tools/llm/llm.py \
	   	  $dec_tgt_emb $dec_src_emb \
		  --ignore-exact-words \
		  --num-neighbors $llm_nn \
		  < $dec_tgt_emb \
		  > $dec_mapped_emb.tmp
	fi
	;;
    linear-idt)
	if [ ! -e $dec_mapped_emb ]; then
	    python tools/vecmap/map_embeddings.py \
		   --init_identical --orthogonal --cuda \
		   $dec_tgt_emb $dec_src_emb \
		   $dec_mapped_emb.tmp $dec_src_emb.$direction.linear-idt
	    rm $dec_src_emb.$direction.linear-idt
	fi
	;;

    llm-v2)
	if [ ! -e $dec_mapped_emb ]; then
	    dec_src_emb_aligned=$dec_src_emb.${src_domain}2${tgt_domain}.aligned
	    dec_tgt_emb_aligned=$dec_tgt_emb.${src_domain}2${tgt_domain}.aligned

	    script_dir=tools/llm-v2
	    python $script_dir/scripts/align.py \
		--src-input-path $dec_tgt_emb \
		--trg-input-path $dec_src_emb \
		--src-aligned-path $dec_tgt_emb_aligned \
		--trg-aligned-path $dec_src_emb_aligned 

	    llm_map \
		   $dec_tgt_emb_aligned $dec_src_emb_aligned \
		   --config-path $script_dir/configs/bispace.base.json \
		   --config-log $script_dir/configs/$direction.json \
		   --detail-path $script_dir/$direction.npz \
		   -o reconstructor.type=\"bispace\" \
		   -o mapper.type=\"bispace\" \
		   -o nn_searcher.src.type=\"csls-topk\"  \
		   -o nn_searcher.input.type=\"csls-topk\"  \
		   -o nn_searcher.src.csls=$llm_nn \
		   -o nn_searcher.input.csls=$llm_nn \
		   -o nn_searcher.src.topk=$llm_nn \
		   -o nn_searcher.input.topk=$llm_nn \
		   < $dec_tgt_emb > $dec_mapped_emb.tmp
	    rm $dec_src_emb_aligned
	    rm $dec_tgt_emb_aligned
	fi
	;;


    * ) echo "invalid embedding type: $emb_type"
        exit 1 ;;
esac
if [ ! -e $dec_mapped_emb ] && [ -e $dec_mapped_emb.tmp ]; then

    # The embeddings of the special words are kept even after mapping.
    sed -n 2,$(($N_SPECIAL_WORDS+1))p $dec_src_emb  > $dec_mapped_emb # Copy the special word embeddings from the target (model's embs)
    sed -n 2,$(wc -l $dec_mapped_emb.tmp | cut -f1 -d ' ')p $dec_mapped_emb.tmp | grep -v '<pad>' | grep -v '</s>' | grep -v '<unk>'  | grep -v '<s>'   >> $dec_mapped_emb  # Gather the tmpfile of mapped embeddings if the word is not a special word.
    n_vocab=$(wc -l $dec_mapped_emb | cut -f1 -d ' ')
    n_dims=$(head -n1 $dec_mapped_emb.tmp | cut -f2 -d ' ') # Count the number of total words again, and insert it to the head.
    sed -i "1i $n_vocab $n_dims" $dec_mapped_emb
    rm $dec_mapped_emb.tmp
fi



# Delete if the projection was failed
if [ ! -s $enc_mapped_emb ]; then
    rm $enc_mapped_emb
    echo "Failed to map embeddings to $enc_mapped_emb."
fi

if [ ! -s $dec_mapped_emb ]; then
    rm $dec_mapped_emb
    echo "Failed to map embeddings to $dec_mapped_emb."
fi
