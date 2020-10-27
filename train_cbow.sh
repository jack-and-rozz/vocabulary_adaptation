#!/bin/bash
echo "Running '$0 $1 $2'..."
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
root_dir=$(pwd)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)

is_valid=$(validate_mode $mode $task)
if [ -n "$is_valid" ]; then
    echo $is_valid
    exit 1
fi

src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)


if [[ $mode =~ \.$outdomain_ext\.v_${tgt_domain} ]]; then
    spm_model_dir=$tgt_data_dir
    data_dir=$src_data_dir
elif [[ $mode =~ \.$outdomain_ext\. ]]; then
    spm_model_dir=$src_data_dir
    data_dir=$src_data_dir
elif [[ $mode =~ \.$indomain_ext\. ]]; then
    spm_model_dir=$src_data_dir
    data_dir=$src_data_dir
elif [[ $mode =~ \.$finetune_ext\.v_${src_domain} ]]; then
    spm_model_dir=$src_data_dir
    data_dir=$tgt_data_dir
elif [[ $mode =~ \.$finetune_ext\.v_${tgt_domain} ]]; then
    spm_model_dir=$tgt_data_dir
    data_dir=$tgt_data_dir
elif [[ $mode =~ \.$vocabadapt_ext\. ]]; then
    spm_model_dir=$tgt_data_dir
    data_dir=$tgt_data_dir
elif [[ $mode =~ .${multidomain_ext}.domainweighting ]]; then
    spm_model_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
    data_dir=$spm_model_dir
elif [[ $mode =~ .${multidomain_ext}.domainmixing ]]; then
    spm_model_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
    data_dir=$spm_model_dir
elif [[ $mode =~ .(${backtranslation_ext}_[a-z]+)\. ]]; then
    bt_type=${BASH_REMATCH[1]}
    spm_model_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain $bt_type)
    data_dir=$spm_model_dir
else
    echo "Invalid mode: $mode"
    exit 1
fi


if [ $task != translation ]; then 
    src_data=$spm_model_dir/train.flat
    src_output=$spm_model_dir/word2vec.${src_lang}.${emb_size}d
else
    src_data=$spm_model_dir/monolingual.${src_lang}
    tgt_data=$spm_model_dir/monolingual.${tgt_lang}
    src_output=$spm_model_dir/word2vec.${src_lang}.${emb_size}d
    tgt_output=$spm_model_dir/word2vec.${tgt_lang}.${emb_size}d
fi

# word2vec raises an error when the output path is too long...
timestamp=$(date "+%s")
tmp_src_output=/tmp/$timestamp.${src_lang}.${emb_size}d
tmp_tgt_output=/tmp/$timestamp.${tgt_lang}.${emb_size}d


if [ ! -e $src_output ]; then
    echo "Training word2vec $src_output from $src_data..."
    ./tools/word2vec/word2vec -size ${emb_size} \
    			      -train $src_data \
    			      -output $tmp_src_output \
    			      -save-vocab $tmp_src_output.vocab \
    			      -min-count $w2v_mincount & 
fi
if [ ! -e $tgt_output ]; then
    if [ ! -z $tgt_data ]; then
	echo "Training word2vec $tgt_output from $tgt_data..."
	./tools/word2vec/word2vec -size ${emb_size} \
				  -train $tgt_data \
				  -output $tmp_tgt_output \
				  -save-vocab $tmp_tgt_output.vocab \
				  -min-count $w2v_mincount &
    fi
fi
wait
if [ ! -e $src_output ]; then
    mv $tmp_src_output $src_output
    mv $tmp_src_output.vocab $src_output.vocab
fi
if [ ! -e $tgt_output ]; then
    mv $tmp_tgt_output $tgt_output
    mv $tmp_tgt_output.vocab $tgt_output.vocab
fi

n_special_words=4 # Defined in fairseq/data/dictionary.py .
src_dict=$spm_model_dir/dict.${src_lang}.txt
tgt_dict=$spm_model_dir/dict.${tgt_lang}.txt


if [ -z "$src_vocab_size" ]; then
    src_vocab_size=$n_vocab_default
fi
if [ -z "$tgt_vocab_size" ]; then
    tgt_vocab_size=$n_vocab_default
fi

if [ ! -e $src_dict ] && [ -e $src_output.vocab ]; then
   n_words=$(($src_vocab_size-$n_special_words)) # tgt_vocab_size indicates the vocabulary size in the **target domain** (TODO: two meanings of 'src' are misleading...).
   echo "Creating vocabulary file (#tokens=$n_words) from '${src_output}.vocab' to '$src_dict'."
   sed -n "2,$((n_words+1))p" $src_output.vocab > $src_dict
fi

if [ ! -e $tgt_dict ] && [ ! -z ${tgt_lang} ] && [ -e $tgt_output.vocab ]; then
    n_words=$(($tgt_vocab_size-$n_special_words))
    echo "Creating vocabulary file (#tokens=$n_words) from '${tgt_output}.vocab' to '$tgt_dict'."
    sed -n "2,$((n_words+1))p" $tgt_output.vocab > $tgt_dict
fi


if [ $spm_model_dir != $data_dir ]; then
    if [ ! -e $data_dir/word2vec.$src_lang ]; then
	ln -sf \
	   $root_dir/$spm_model_dir/word2vec.$src_lang.${emb_size}d \
	   $data_dir/word2vec.$src_lang.${emb_size}d
	ln -sf \
	   $root_dir/$spm_model_dir/word2vec.$src_lang.${emb_size}d.vocab \
	   $data_dir/word2vec.$src_lang.${emb_size}d.vocab
    fi
    if [ ! -e $tgt_data_dir/word2vec.$tgt_lang ] && [ ! -z $tgt_data ]; then
	ln -sf \
	   $root_dir/$spm_model_dir/word2vec.$tgt_lang.${emb_size}d \
	   $data_dir/word2vec.$tgt_lang.${emb_size}d
	ln -sf \
	   $root_dir/$spm_model_dir/word2vec.$tgt_lang.${emb_size}d.vocab \
	   $data_dir/word2vec.$tgt_lang.${emb_size}d.vocab
    fi

    if [ ! -e $data_dir/dict.$src_lang.txt ]; then
	ln -sf \
	   $root_dir/$spm_model_dir/dict.$src_lang.txt \
	   $data_dir/dict.$src_lang.txt
    fi
    if [ ! -e $tgt_data_dir/dict.$tgt_lang ] && [ ! -z $tgt_data ]; then
	ln -sf \
	   $root_dir/$spm_model_dir/dict.$tgt_lang.txt \
	   $data_dir/dict.$tgt_lang.txt
    fi

fi
