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
if [ -n "$is_valid" ]; then
    echo $is_valid
    exit 1
fi

root_dir=$(pwd)
size=$(parse_size $mode)
ft_size=$size
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)
tgt_data_dir_wd=$(get_data_dir $mode $tgt_domain_wd)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

bt_aug_data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_aug)
bt_ft_data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_ft)
bt_va_data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_va)

train_prefix=train.$size

src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)

langs=($src_lang $tgt_lang)
src_spm_domain=$(parse_spm_domain $mode src)
src_spm_mono_size=$(parse_spm_mono_size $mode src)
tgt_spm_domain=$(parse_spm_domain $mode tgt)
tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)


if [[ $mode =~ _${sp_suffix} ]]; then
    # augmentation_model=$ckpt_root/${src_domain}${src_vocab_size}${direction_tok}${tgt_domain}${tgt_vocab_size}.${backtranslation_ext}_aug.v_${src_domain}${src_vocab_size}_${src_spm_mono_size}.$size
    augmentation_model=$ckpt_root/${src_domain}${src_vocab_size}${direction_tok}${tgt_domain}${tgt_vocab_size}.${backtranslation_ext}_aug.v_${src_domain}${src_vocab_size}_all.$size
    bt_outputs=$augmentation_model/tests/$tgt_domain_wd.outputs 
    bt_inputs=$augmentation_model/tests/$tgt_domain_wd.inputs 
else
    exit 1
    # augmentation_model=$ckpt_root/${src_domain}.${backtranslation_ext}_aug.$v_{src_domain}.$size
    # bt_outputs=$augmentation_model/tests/$tgt_domain.outputs 
    # bt_inputs=$augmentation_model/tests/$tgt_domain.inputs 
fi

#############################################
#       For data augmentation (bt_aug)
#############################################

# Make symbolic links to train/dev sets of the source domain, and to the test set of the target domain. Pre-trained word embeddings and subword tokenization of the source domain are also linked.

if [[ $mode =~ ${backtranslation_ext}_aug ]] && [ ! -e $bt_aug_data_dir/train.$src_lang ]; then
    echo "Preparing the dataset for training a back-translation model ($augmentation_model) to '$bt_aug_data_dir'."
    if [ ! -e $bt_aug_data_dir ]; then
	mkdir -p $bt_aug_data_dir
    fi
    src_spm_dir=$src_data_dir
    monolingual_file=mono.ft${ft_size}

    for lang in ${langs[@]}; do
	ln -sf $root_dir/$src_spm_dir/word2vec.$lang.${emb_size}d $bt_aug_data_dir
	ln -sf $root_dir/$src_spm_dir/word2vec.$lang.${emb_size}d.vocab $bt_aug_data_dir
	ln -sf $root_dir/$src_spm_dir/dict.$lang.txt $bt_aug_data_dir

	ln -sf $root_dir/$src_data_dir/train.$lang $bt_aug_data_dir
	ln -sf $root_dir/$src_data_dir/train.all.$lang $bt_aug_data_dir
	ln -sf $root_dir/$src_data_dir/dev.$lang $bt_aug_data_dir

	# Process the training set in target domain as a monolingual corpus for data augmentation by back-translation.
	if [[ $src_domain =~ _${sp_suffix} ]]; then
	    # Encode the (word-level) monolingual corpus in the target domain by the SentencePiece trained in the src domain.
	    ln -sf $root_dir/$src_data_dir/spm.$lang.model $bt_aug_data_dir
	    ln -sf $root_dir/$src_data_dir/spm.$lang.vocab $bt_aug_data_dir
	    if [ ! -e $bt_aug_data_dir/test.$lang ]; then
		spm_encode --model $bt_aug_data_dir/spm.$lang.model \
			   --output $bt_aug_data_dir/mono.ft${ft_size}.$lang \
			   < $tgt_data_dir_wd/$monolingual_file.$lang &
	    fi
	else
	    ln -sf \
	       $root_dir/$tgt_data_dir_wd/$monolingual_file.$lang \
	       $bt_aug_data_dir/mono.ft${ft_size}.$lang
	fi
    done
    wait
    exit 1
fi
wait

if [ ! -e $augmentation_model/checkpoints/checkpoint_best.pt ]; then
    echo "$augmentation_model/checkpoints/checkpoint_best.pt was not found. Train a model for data augmentation in advance."
    exit 1
fi

if [ ! -e $bt_outputs ]; then
    echo "Error: $bt_outputs was not found. Train a model for back-translation in the source domain and translate a target-domain monolingual corpus."
    exit 1
fi

#############################################################################
#   For fine-tuning by the augmented data with source domain vocab (bt_ft)
#############################################################################

# if [ ! -e $bt_ft_data_dir/$train_prefix.$src_lang ] && [ $src_vocab_size == $tgt_vocab_size ]; then

# echo "$bt_ft_data_dir/$train_prefix."
# echo $src_spm_domain $src_spm_mono_size
# echo $tgt_spm_domain $tgt_spm_mono_size
# echo $src_data_dir
# echo $tgt_data_dir
# exit 1

if [[ $mode =~ \.${backtranslation_ext}_ft\. ]] && [ ! -e $bt_ft_data_dir/$train_prefix.$src_lang ]; then
# if [ 1 == 1 ]; then # for debug
    echo "Preparing the joined dataset to $bt_ft_data_dir..."
    if [ ! -e $bt_ft_data_dir ]; then
	mkdir -p $bt_ft_data_dir
    fi

    for lang in ${langs[@]}; do
	ln -sf $root_dir/$src_data_dir/word2vec.$lang.${emb_size}d $bt_ft_data_dir
	ln -sf $root_dir/$src_data_dir/word2vec.$lang.${emb_size}d.vocab $bt_ft_data_dir
	ln -sf $root_dir/$src_data_dir/dict.$lang.txt $bt_ft_data_dir
	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	    ln -sf $root_dir/$src_data_dir/spm.$lang.model $bt_ft_data_dir
	    ln -sf $root_dir/$src_data_dir/spm.$lang.vocab $bt_ft_data_dir
	fi
    done
    
    # Combine the fine-tuning set of the target domain and the outputs of the trained back-translation model.
    if [ ! -e $bt_ft_data_dir/$train_prefix.$src_lang ]; then
	# First, copy or encode the fine-tuning set in the target domain.
	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	    for lang in ${langs[@]}; do
		spm_encode --model $bt_ft_data_dir/spm.$lang.model \
			   --output $bt_ft_data_dir/$train_prefix.$lang  \
			   < $tgt_data_dir_wd/$train_prefix.$lang &
	    done
	else
	    for lang in ${langs[@]}; do
		cp $tgt_data_dir/$train_prefix.$lang $bt_ft_data_dir &
	    done
	fi

	# Second, copy or encode the outputs of the back-translation model and the monolingual corpus.
	if [ ! -e mono.bt.$src_lang ]; then
	    if [[ $tgt_domain =~ _${sp_suffix} ]]; then
		if [ ! -e $bt_ft_data_dir/mono.bt.$src_lang ]; then
		    spm_encode --model $bt_ft_data_dir/spm.$src_lang.model \
			       --output $bt_ft_data_dir/mono.bt.$src_lang  \
			       < $bt_outputs &
		fi

		if [ ! -e $bt_ft_data_dir/mono.bt.$tgt_lang ]; then
		    spm_encode --model $bt_ft_data_dir/spm.$tgt_lang.model \
			       --output $bt_ft_data_dir/mono.bt.$tgt_lang  \
			       < $bt_inputs &
		fi
	    else
		cat $bt_outputs >> $bt_ft_data_dir/mono.bt.$src_lang &
		cat $bt_inputs >> $bt_ft_data_dir/mono.bt.$tgt_lang &
	    fi
	    wait
	fi
	wait 

    fi
    cat $bt_ft_data_dir/mono.bt.$src_lang \
	>> $bt_ft_data_dir/$train_prefix.$src_lang & 
    cat $bt_ft_data_dir/mono.bt.$tgt_lang \
	>> $bt_ft_data_dir/$train_prefix.$tgt_lang & 
    wait 


    # Copy or encode the dev/test sets in the target domain.
    dtypes=(dev test)
    if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	for dtype in ${dtypes[@]}; do
	    for lang in ${langs[@]}; do
		if [ ! -e $bt_ft_data_dir/$dtype.$lang ]; then
		    spm_encode --model $bt_ft_data_dir/spm.$lang.model \
			       --output $bt_ft_data_dir/$dtype.$lang  \
			       < $tgt_data_dir_wd/$dtype.$lang &
		fi
	    done
	done
    else
	for lang in ${langs[@]}; do
	    ln -sf $root_dir/$tgt_data_dir/dev.$lang $bt_ft_data_dir
	    ln -sf $root_dir/$tgt_data_dir/test.$lang $bt_ft_data_dir
	done
    fi
    if [ $size == all ]; then
	ln -sf train.$src_lang $bt_ft_data_dir/train.all.$src_lang
	ln -sf train.$tgt_lang $bt_ft_data_dir/train.all.$tgt_lang
    fi
    wait 
fi

#############################################################################
#   For fine-tuning by the augmented data with target domain vocab (back-translation_va)
#############################################################################

# if [ 1 == 1 ]; then
if [[ $mode =~ \.${backtranslation_ext}_va\. ]] && [ ! -e $bt_va_data_dir/$train_prefix.$src_lang ]; then

    echo "Preparing the joined dataset to $bt_va_data_dir..."
    if [ ! -e $bt_va_data_dir ]; then
	mkdir -p $bt_va_data_dir
    fi
    for lang in ${langs[@]}; do
	ln -sf $root_dir/$tgt_data_dir/word2vec.$lang.${emb_size}d $bt_va_data_dir
	ln -sf $root_dir/$tgt_data_dir/word2vec.$lang.${emb_size}d.vocab $bt_va_data_dir
	ln -sf $root_dir/$tgt_data_dir/dict.$lang.txt $bt_va_data_dir
	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	    ln -sf $root_dir/$tgt_data_dir/spm.$lang.model $bt_va_data_dir
	    ln -sf $root_dir/$tgt_data_dir/spm.$lang.vocab $bt_va_data_dir
	fi
    done

    if [ ! -e $bt_va_data_dir/$train_prefix.$src_lang ]; then
	# First, copy or encode the fine-tuning set in the target domain.
	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	    for lang in ${langs[@]}; do
		spm_encode --model $bt_va_data_dir/spm.$lang.model \
			   --output $bt_va_data_dir/$train_prefix.$lang  \
			   < $tgt_data_dir_wd/$train_prefix.$lang &
	    done
	else
	    for lang in ${langs[@]}; do
		cp $tgt_data_dir/$train_prefix.$lang $bt_va_data_dir &
	    done
	fi
	wait 


	# Second, copy or encode the outputs of the back-translation model and the monolingual corpus.
	if [ ! -e mono.bt.$src_lang ]; then
	    if [[ $tgt_domain =~ _${sp_suffix} ]]; then
		spm_encode --model $bt_va_data_dir/spm.$src_lang.model \
			   --output $bt_va_data_dir/mono.bt.$src_lang  \
			   < $bt_outputs &

		spm_encode --model $bt_va_data_dir/spm.$tgt_lang.model \
			   --output $bt_va_data_dir/mono.bt.$tgt_lang  \
			   < $bt_inputs &

	    else
		cat $bt_outputs >> $bt_va_data_dir/mono.bt.$src_lang &
		cat $bt_inputs >> $bt_va_data_dir/mono.bt.$tgt_lang &
	    fi
	    wait
	fi
    fi
    cat $bt_va_data_dir/mono.bt.$src_lang \
	>> $bt_va_data_dir/$train_prefix.$src_lang & 
    cat $bt_va_data_dir/mono.bt.$tgt_lang \
	>> $bt_va_data_dir/$train_prefix.$tgt_lang & 
    wait 


    # Copy or encode the dev/test sets in the target domain.
    dtypes=(dev test)
    if [[ $tgt_domain =~ _${sp_suffix} ]]; then
	for dtype in ${dtypes[@]}; do
	    for lang in ${langs[@]}; do
		if [ ! -e $bt_va_data_dir/$dtype.$lang ]; then
		    spm_encode --model $bt_va_data_dir/spm.$lang.model \
			       --output $bt_va_data_dir/$dtype.$lang  \
			       < $tgt_data_dir_wd/$dtype.$lang &
		fi
	    done
	done
    else
	for lang in ${langs[@]}; do
	    ln -sf $root_dir/$tgt_data_dir/dev.$lang $bt_va_data_dir
	    ln -sf $root_dir/$tgt_data_dir/test.$lang $bt_va_data_dir
	done
    fi
    wait 
fi


# bt_va_enc_data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_va_enc)
# bt_va_dec_data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_va_dec)


# # Create data encoded by source domain's SentencePiece for ablation tests.
# if [[ $mode =~ \.${backtranslation_ext}_va_enc\. ]] && [ ! -e $bt_va_enc_data_dir/$train_prefix.$src_lang ]; then
#     echo "Preparing the joined dataset to $bt_va_enc_data_dir..."
#     if [ ! -e $bt_va_enc_data_dir ]; then
# 	mkdir -p $bt_va_enc_data_dir
#     fi
#     for lang in ${langs[@]}; do
# 	ln -sf $root_dir/$tgt_data_dir/word2vec.$lang.${emb_size}d $bt_va_enc_data_dir
# 	ln -sf $root_dir/$tgt_data_dir/word2vec.$lang.${emb_size}d.vocab $bt_va_enc_data_dir
# 	ln -sf $root_dir/$tgt_data_dir/dict.$lang.txt $bt_va_enc_data_dir
# 	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
# 	    ln -sf $root_dir/$tgt_data_dir/spm.$lang.model $bt_va_enc_data_dir
# 	    ln -sf $root_dir/$tgt_data_dir/spm.$lang.vocab $bt_va_enc_data_dir
# 	fi
#     done

#     if [ ! -e $bt_va_enc_data_dir/$train_prefix.$src_lang ]; then
# 	# First, copy or encode the fine-tuning set in the target domain.
# 	if [[ $tgt_domain =~ _${sp_suffix} ]]; then
# 	    for lang in ${langs[@]}; do
# 		spm_encode --model $bt_va_enc_data_dir/spm.$lang.model \
# 			   --output $bt_va_enc_data_dir/$train_prefix.$lang  \
# 			   < $tgt_data_dir_wd/$train_prefix.$lang &
# 	    done
# 	else
# 	    for lang in ${langs[@]}; do
# 		cp $tgt_data_dir/$train_prefix.$lang $bt_va_enc_data_dir &
# 	    done
# 	fi
# 	wait 


# 	# Second, copy or encode the outputs of the back-translation model and the monolingual corpus.
# 	if [ ! -e mono.bt.$src_lang ]; then
# 	    if [[ $tgt_domain =~ _${sp_suffix} ]]; then
# 		spm_encode --model $bt_va_enc_data_dir/spm.$src_lang.model \
# 			   --output $bt_va_enc_data_dir/mono.bt.$src_lang  \
# 			   < $bt_outputs &

# 		spm_encode --model $bt_va_enc_data_dir/spm.$tgt_lang.model \
# 			   --output $bt_va_enc_data_dir/mono.bt.$tgt_lang  \
# 			   < $bt_inputs &

# 	    else
# 		cat $bt_outputs >> $bt_va_enc_data_dir/mono.bt.$src_lang &
# 		cat $bt_inputs >> $bt_va_enc_data_dir/mono.bt.$tgt_lang &
# 	    fi
# 	    wait
# 	fi
#     fi
#     cat $bt_va_enc_data_dir/mono.bt.$src_lang \
# 	>> $bt_va_enc_data_dir/$train_prefix.$src_lang & 
#     cat $bt_va_enc_data_dir/mono.bt.$tgt_lang \
# 	>> $bt_va_enc_data_dir/$train_prefix.$tgt_lang & 
#     wait 


#     # Copy or encode the dev/test sets in the target domain.
#     dtypes=(dev test)
#     if [[ $tgt_domain =~ _${sp_suffix} ]]; then
# 	for dtype in ${dtypes[@]}; do
# 	    for lang in ${langs[@]}; do
# 		if [ ! -e $bt_va_enc_data_dir/$dtype.$lang ]; then
# 		    spm_encode --model $bt_va_enc_data_dir/spm.$lang.model \
# 			       --output $bt_va_enc_data_dir/$dtype.$lang  \
# 			       < $tgt_data_dir_wd/$dtype.$lang &
# 		fi
# 	    done
# 	done
#     else
# 	for lang in ${langs[@]}; do
# 	    ln -sf $root_dir/$tgt_data_dir/dev.$lang $bt_va_enc_data_dir
# 	    ln -sf $root_dir/$tgt_data_dir/test.$lang $bt_va_enc_data_dir
# 	done
#     fi
#     wait 
# fi


