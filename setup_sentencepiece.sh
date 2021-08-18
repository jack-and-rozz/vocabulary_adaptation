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

src_domain=$(remove_tok_suffix $src_domain)
tgt_domain=$(remove_tok_suffix $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 
langs=($src_lang $tgt_lang)

size=$(parse_size $mode)
files=(train.$size dev test)

src_truecased_dir=$(get_data_dir $mode $src_domain)
tgt_truecased_dir=$(get_data_dir $mode $tgt_domain)
src_truecased_sp_dir=$(get_data_dir $mode ${src_domain}_${sp_suffix})
tgt_truecased_sp_dir=$(get_data_dir $mode ${tgt_domain}_${sp_suffix})

src_spm_domain=$(parse_spm_domain $mode src)
src_spm_mono_size=$(parse_spm_mono_size $mode src)
tgt_spm_domain=$(parse_spm_domain $mode tgt)
tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

if [[ $mode =~ \.$outdomain_ext\.v_${tgt_domain} ]]; then
    spm_mono_dir=$tgt_truecased_dir
    spm_model_dir=$tgt_truecased_sp_dir
    spm_mono_size=$tgt_spm_mono_size
    truecased_dir=$src_truecased_dir
    truecased_sp_dir=$src_truecased_sp_dir
elif [[ $mode =~ \.${vocabadapt_ext}\. ]] || [[ $mode =~ \.${vocabadapt_ext}_enc\. ]] || [[ $mode =~ \.${vocabadapt_ext}_dec\. ]]; then
    spm_mono_dir=$tgt_truecased_dir
    spm_model_dir=$tgt_truecased_sp_dir
    spm_mono_size=$tgt_spm_mono_size
    truecased_dir=$tgt_truecased_dir
    truecased_sp_dir=$tgt_truecased_sp_dir
else
    spm_mono_dir=$src_truecased_dir
    spm_model_dir=$src_truecased_sp_dir
    spm_mono_size=$src_spm_mono_size
    truecased_dir=$tgt_truecased_dir
    truecased_sp_dir=$tgt_truecased_sp_dir
fi

if [ $spm_mono_size == all ]; then
    spm_mono_file=train.all
    cbow_mono_file=$spm_mono_file
elif [[ $spm_mono_size =~ ([0-9]+k)monoCBoWonly ]]; then
    ft_size=${BASH_REMATCH[1]}
    spm_mono_file=train.$ft_size
    cbow_mono_file=mono.ft${ft_size}+ft
elif [[ $spm_mono_size =~ ([0-9]+k)mono ]]; then
    ft_size=${BASH_REMATCH[1]}
    spm_mono_file=mono.ft${ft_size}+ft
    cbow_mono_file=$spm_mono_file
elif [[ $spm_mono_size =~ ([0-9]+k) ]]; then
    spm_mono_file=train.${spm_mono_size}
    cbow_mono_file=$spm_mono_file
    if [ ! -e $spm_mono_dir/$spm_mono_file.$src_lang ]; then
	python scripts/random_pickup.py \
	       $spm_mono_dir/train.$src_lang \
	       $spm_mono_dir/train.$tgt_lang \
	       $spm_mono_size \
	       --seed $random_seed
    fi

else
    echo "Error (spm_mono_size=$spm_mono_size)"
    exit 1
fi


if [ -z $(which spm_encode) ]; then
    echo "spm_encode was not found. Install sentencepiece following 'https://github.com/google/sentencepiece'."
    exit 1
fi

if [[ $mode =~ multidomain ]]; then
    # "Subword tokenization for multidomain data is done in ./setup_multidomain_data.sh."
    exit 1
fi

# Directory to store train/dev/test sets encoded by spm.
if [ ! -e $truecased_sp_dir ]; then
    mkdir -p $truecased_sp_dir
fi

# Directory to store a SentencePiece model 
if [ ! -e $spm_model_dir ]; then
    mkdir -p $spm_model_dir
fi

if [ $task == translation ]; then
    for lang in ${langs[@]}; do
	if [ ! -e $spm_model_dir/spm.$lang.model ]; then
	    echo "training $spm_model_dir/spm.$lang from $spm_mono_dir/${spm_mono_file}.$lang..."
	    spm_train --vocab_size ${tgt_vocab_size} \
		      --model_prefix $spm_model_dir/spm.$lang \
		      --unk_surface $unk_surface \
		      --input_sentence_size 2000000 \
		      --shuffle_input_sentence \
		      --hard_vocab_limit=false \
		      --model_type=$spm_model_type \
		      --input $spm_mono_dir/${spm_mono_file}.$lang &
	fi
    done
else
    if [ ! -e $spm_model_dir/spm.$src_lang.model ]; then
	echo "training $spm_model_dir/spm.$lang from $spm_mono_dir/${spm_mono_file}.$src_lang..."
	spm_train --vocab_size ${tgt_vocab_size} \
		  --model_prefix $spm_model_dir/spm.$src_lang \
		  --unk_surface $unk_surface \
		  --input_sentence_size 2000000 \
		  --shuffle_input_sentence \
		  --hard_vocab_limit=false \
		  --model_type=$spm_model_type \
		  --input $spm_mono_dir/${spm_mono_file}.$src_lang &
    fi
    ln -sf spm.$src_lang.model $spm_model_dir/spm.$tgt_lang.model 
    ln -sf spm.$src_lang.vocab $spm_model_dir/spm.$tgt_lang.vocab
fi

wait 
for lang in ${langs[@]}; do
    if [ ! -e $truecased_sp_dir/spm.$lang.model ]; then
	ln -sf $root_dir/$spm_model_dir/spm.$lang.model $truecased_sp_dir/spm.$lang.model
	ln -sf $root_dir/$spm_model_dir/spm.$lang.vocab $truecased_sp_dir/spm.$lang.vocab
    fi
done



for file in ${files[@]}; do
    for lang in ${langs[@]}; do
	if [ ! -e $truecased_sp_dir/$file.$lang ]; then
	    echo "encoding '$truecased_dir/$file.$lang' to '$truecased_sp_dir/$file.$lang'..."

	    spm_encode --model $truecased_sp_dir/spm.$lang.model \
		       --output $truecased_sp_dir/$file.$lang \
		       < $truecased_dir/$file.$lang &
	fi
    done
done

for lang in ${langs[@]}; do
    if [ -e $truecased_sp_dir/train.all.$lang ]; then
	ln -sf train.all.$lang $truecased_sp_dir/train.$lang 
    fi
done


for lang in ${langs[@]}; do
    if [ $spm_model_dir == $truecased_sp_dir ]; then
	if [ ! -e $truecased_sp_dir/monolingual.$lang ]; then
	    spm_encode --model $spm_model_dir/spm.$lang.model \
		       --output $truecased_sp_dir/monolingual.$lang \
		       < ${spm_mono_dir}/$cbow_mono_file.$lang &
	fi
    else
	if [ ! -e $truecased_sp_dir/monolingual.$lang ]; then
	    ln -sf $root_dir/$spm_model_dir/monolingual.$lang $truecased_sp_dir/monolingual.$lang
	fi
    fi
done
wait
