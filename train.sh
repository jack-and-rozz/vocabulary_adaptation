#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task [train_steps] [update_freq]"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi


. ./const.sh $mode $task

mode=$1
task=$2
train_steps_specified=$3
_update_freq=$4

is_valid=$(validate_mode $mode $task)
if [ -n "$is_valid" ]; then
    echo $is_valid
    exit 1
fi

model_dir=$(get_model_dir $ckpt_root $mode)
architecture=transformer_finetuning
criterion=label_smoothed_cross_entropy

size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task)
tgt_lang=$(get_tgt_lang $tgt_domain $task)
emb_type=$(parse_emb_type $mode)
multidomain_type=$(parse_multidomain_type $mode)
fixed=$(parse_fixed $mode)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)

src_spm_domain=$(parse_spm_domain $mode src)
src_spm_mono_size=$(parse_spm_mono_size $mode src)
tgt_spm_domain=$(parse_spm_domain $mode tgt)
tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)


echo "task="$task
echo "size="$size
echo "src_domain="$src_domain
echo "tgt_domain="$tgt_domain
echo "src_lang="$src_lang
echo "tgt_lang="$tgt_lang
echo "emb_type="$emb_type
echo "multidomain_type="$multidomain_type
echo "fixed="$fixed

train_steps=$train_steps_specified
if [ -z $train_steps ]; then
    train_steps=$(eval echo '$train_steps_'$size)
fi
if [ -z $train_steps ]; then
    train_steps=$(eval echo '$train_steps_'$task)
fi
if [ -z $train_steps ]; then
    train_steps=$train_steps_default
fi

case $mode in
    #####################################
    ##        Training / Pretraining
    #####################################
    # Train a model in the src domain w/ src domain vocab.
    # The vocabulary is constructed from large source-domain parallel data.
    # - Out-domain (training)
    # - FT-srcV (pre-training)
    # - VA (pre-training)
    # - BT (pre-training)
    *.${outdomain_ext}.${size})
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size

	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;

    # Train a model in the tgt domain w/ tgt domain vocab.
    # The vocabulary is constructed from 
    # 1. small target-domain parallel data 
    # and
    # 2. simulated target-domain monolingual data by splitting large parallel data.
    # - In-domain (training)
    *.${indomain_ext}.*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size

	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;


    # Train a model in the src domain w/ tgt domain vocab.
    # - FT-tgtV (training in the src domain)
    *.${outdomain_ext}.v_${tgt_domain}*)
	data_dir=$(get_data_dir $mode $src_domain)
	data=$data_dir/fairseq.$size

	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;

    ########################################
    ###     Fine-tuning
    ########################################

    # Domain adaptation by fine-tuning w/ src domain vocab.
    # - FT-srcV (fine-tuning)
    *${direction_tok}*.${finetune_ext}.v_${src_domain}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size

	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ] ; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}"
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;

    # Domain adaptation by fine-tuning w/ tgt domain vocab.
    # - FT-tgtV (fine-tuning)

    *${direction_tok}*.${finetune_ext}.v_${tgt_domain}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
 	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)
	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.v_${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}.all

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}"
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;

    # Domain adaptation by fine-tuning w/ vocabulary adaptation.
    # - VA (fine-tuning)
    *${direction_tok}*.${vocabadapt_ext}.*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size

	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all
	if [ ! -z $fixed ]; then
	    restore_root=$restore_root.fixed
	fi
	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

	enc_emb_path=$restore_root/embeddings/encoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type
	dec_emb_path=$restore_root/embeddings/decoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --override-embeddings \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi


	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}"
	;;

    *${direction_tok}*.${vocabadapt_ext}_enc.*)
	tgt_data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir=$tgt_data_dir.enc
	data=$data_dir/fairseq.$size

	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all
	if [ ! -z $fixed ]; then
	    restore_root=$restore_root.fixed
	fi
	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)
	enc_emb_path=$restore_root/embeddings/encoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type
	dec_emb_path=$restore_root/embeddings/decoder.indomain

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --override-embeddings \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}"
	;;

    *${direction_tok}*.${vocabadapt_ext}_dec.*)
	tgt_data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir=$tgt_data_dir.dec
	data=$data_dir/fairseq.$size

	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all
	if [ ! -z $fixed ]; then
	    restore_root=$restore_root.fixed
	fi
	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

	enc_emb_path=$restore_root/embeddings/encoder.indomain
	dec_emb_path=$restore_root/embeddings/decoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --override-embeddings \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}"
	;;

    ########################################
    ###     Multidomain-learning
    ########################################

    # *${direction_tok}*.mdl.domainweighting.*)
    # 	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
    # 	data=$data_dir/fairseq.$size
    # 	train_options="--max-update $train_steps"
    # 	task_options="--task ${fairseq_task} \
    # 		      --source-lang ${src_lang} \
    # 		      --target-lang ${tgt_lang} \
    # 		      --extra-features domain
    # 		     "
    # 	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
    # 	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
    # 	criterion=domain_weighting_lsce
    # 	;;
    
    # - MDL (training)
    *${direction_tok}*.mdl.domainmixing.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	data=$data_dir/fairseq.$size

	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	architecture=transformer_domainmixing
	;;

    ########################################
    ###     Back-translation
    ########################################
    # - BT (training of an augmentation model)
    *${direction_tok}*.${backtranslation_ext}_aug.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    ${backtranslation_ext}_aug)
	data=$data_dir/fairseq.$size

	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${tgt_lang} \
		      --target-lang ${src_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	;;

    # - BT (fine-tuning)
    *${direction_tok}*.${backtranslation_ext}_ft.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    ${backtranslation_ext}_ft)
	data=$data_dir/fairseq.$size

	src_spm_domain=$(parse_spm_domain $mode src)
	src_spm_mono_size=$(parse_spm_mono_size $mode src)
	tgt_spm_domain=$(parse_spm_domain $mode tgt)
	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

	if [[ $src_spm_domain =~ $src_domain ]]; then
	    restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all
	elif [[ $src_spm_domain =~ $tgt_domain ]]; then
	    restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.v_${tgt_spm_domain}_${tgt_spm_mono_size}.all
	fi


	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	enc_emb_path=$data_dir/word2vec.${src_lang}.${emb_size}d
	dec_emb_path=$data_dir/word2vec.${tgt_lang}.${emb_size}d
	;;

    # - BT + VA (fine-tuning)
    *${direction_tok}*.${backtranslation_ext}_va.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    ${backtranslation_ext}_va)
	data=$data_dir/fairseq.$size

	restore_root=$ckpt_root/${src_domain}${src_vocab_size}.${outdomain_ext}.all

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --override-embeddings \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)
	enc_emb_path=$restore_root/embeddings/encoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type
	dec_emb_path=$restore_root/embeddings/decoder.${tgt_domain}${tgt_vocab_size}_${tgt_spm_mono_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type
	;;

    # VA (encoder only) # TODO: remove
    *${direction_tok}*.${backtranslation_ext}_va_enc.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    backtranslation_va)
	src_data_dir=$(get_data_dir $mode $src_domain)

	data=$data_dir/fairseq.only_enc.$size

	if [[ $src_domain =~ _${sp_suffix} ]]; then
	    restore_root=$ckpt_root/$src_domain$src_vocab_size.${baseline_suffix}.all
	else
	    restore_root=$ckpt_root/$src_domain.${baseline_suffix}.all
	fi

	# When extending fine-tuning steps, the environment is not reset.
	if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
	    train_options="--max-update $train_steps \
	    	           --reset-optimizer \
		           --reset-dataloader \
		           --reset-lr-scheduler \
		           --reset-args \
		           --override-embeddings \
		           --restore-file $restore_root/checkpoints/checkpoint_best.pt
	              "
	elif [ $train_steps == -1 ]; then
	    train_options="--max-update $train_steps"
	else 
	    train_options="--max-update $train_steps"
	fi

	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	# Use the source domain's vocabulary and embeddings in the decoder.
	enc_emb_path=$restore_root/embeddings/encoder.${tgt_domain}${tgt_vocab_size}${direction_tok}${src_domain}${src_vocab_size}.$emb_type
	dec_emb_path=$restore_root/embeddings/decoder.indomain
	;;

    * ) echo "invalid mode: $mode"
        exit 1 ;;
esac

if [[ $emb_type =~ llm ]]; then
    llm_nn=$(parse_llm_nn $mode)
    if [[ ! $mode =~ \.${vocabadapt_ext}_dec\. ]]; then
	enc_emb_path=$enc_emb_path.nn${llm_nn}
    fi
    if [[ ! $mode =~ \.${vocabadapt_ext}_enc\. ]]; then
	dec_emb_path=$dec_emb_path.nn${llm_nn}
    fi
fi

if [ $task == translation ];then
    emb_options="$emb_options --encoder-embed-path $enc_emb_path \
       	         --decoder-embed-path $dec_emb_path \
		 --share-decoder-input-output-embed
 		"
else
    emb_options="$emb_options --encoder-embed-path $enc_emb_path \
		 --share-decoder-input-output-embed \
		 --share-all-embeddings
 		"
fi
if [ ! -z $fixed ]; then
    emb_options="$emb_options --disable-training-embeddings"
fi

# Prepare in-domain monolingual data if needed
if [[ $tgt_spm_mono_size =~ mono ]];then
    ./setup_monolingual_data.sh $mode $task
fi

if [[ $tgt_domain =~ _${sp_suffix} ]] && [[ ! $mode =~ $multidomain_ext ]] && [[ ! $mode =~ $backtranslation_ext ]]; then
    ./setup_sentencepiece.sh $mode $task
fi

if [[ $mode =~ ${vocabadapt_ext}_enc ]] || [[ $mode =~ ${vocabadapt_ext}_dec ]]; then
    ./setup_ablation_test.sh $mode $task
fi

if [[ $mode =~ $multidomain_ext ]]; then
    ./setup_multidomain_data.sh $mode $task
fi

if [[ $mode =~ $backtranslation_ext ]]; then
    ./setup_monolingual_data.sh $mode $task
    ./setup_backtranslation_data.sh $mode $task
fi

if [ ! -e $enc_emb_path ]; then
    ./train_cbow.sh $mode $task
    case $mode in
	*.${vocabadapt_ext}.*.*)
	    ./map_embeddings.sh $mode $task
	    ;;

	*.${backtranslation_ext}_va*.*)
	    ./map_embeddings.sh $mode $task
	    ;;
    esac
fi

if [ ! -e $data ]; then
    ./preprocess.sh $mode $task
fi

if [ ! -e $model_dir/tests ];then
    mkdir -p $model_dir/tests
fi
if [ ! -e $model_dir/checkpoints ];then
    mkdir -p $model_dir/checkpoints
fi

if [ ! -e $model_dir/tensorboard ];then
    mkdir -p $model_dir/tensorboard
fi
if [ ! -e $model_dir/embeddings ];then
    mkdir -p $model_dir/embeddings
fi
if [ ! -e $model_dir/subword ];then
    mkdir -p $model_dir/subword
fi

# Link to the subword tokenization model used in the NMT model.
suffixes=(model vocab)

if [ -z $src_dom_spm_dir ]; then
    src_dom_spm_dir=$data_dir
fi
if [ -z $tgt_dom_spm_dir ]; then
    tgt_dom_spm_dir=$data_dir
fi

for suffix in ${suffixes[@]}; do
    if [[ $mode =~ ${sp_suffix} ]]; then
	if [ ! -e $model_dir/subword/spm.$src_lang.$suffix ]; then
	    ln -sf $(pwd)/$src_dom_spm_dir/spm.$src_lang.$suffix \
	       $model_dir/subword/spm.$src_lang.$suffix
	fi
	if [ ! -e $model_dir/subword/spm.$tgt_lang.$suffix ]; then
	    ln -sf $(pwd)/$tgt_dom_spm_dir/spm.$tgt_lang.$suffix \
	       $model_dir/subword/spm.$tgt_lang.$suffix
	fi
    fi
done

# To make baselines strong by following [Hu+, ACL'19].
if [[ $mode =~ opus_ ]]; then
    label_smoothing_factor=0.2
    max_tokens_per_batch=2000
    update_freq=8
    train_options="$train_options --encoder-normalize-before --decoder-normalize-before --clip-norm 0 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0001"
fi
if [ ! -z $_update_freq ]; then
    update_freq=$_update_freq
fi

# For debugging
# echo "Preprocessing is done."
# exit 1

echo "Start training $mode..."
# Start training.
python fairseq/train.py \
       --user-dir ${fairseq_user_dir} \
       --ddp-backend=no_c10d \
       --log-interval 50 --log-format simple \
       --save-dir $model_dir/checkpoints \
       --tensorboard-logdir $model_dir/tensorboard \
       --arch $architecture \
       $data \
       $task_options \
       $emb_options \
       $train_options \
       --max-epoch $max_epoch \
       --max-tokens $max_tokens_per_batch \
       --update-freq $update_freq \
       --num-workers 4 \
       --keep-last-epochs 2 \
       --optimizer adam --adam-betas '(0.9, 0.98)' \
       --lr 1e-03 --min-lr 1e-09   \
       --lr-scheduler inverse_sqrt \
       --warmup-init-lr 1e-07  \
       --warmup-updates 4000   \
       --criterion $criterion  \
       --label-smoothing $label_smoothing_factor   \
       --dropout $dropout_rate \
       --encoder-layers $num_encoder_layers \
       --decoder-layers $num_decoder_layers \
       --encoder-attention-heads $num_encoder_attention_heads \
       --decoder-attention-heads $num_decoder_attention_heads \
       --encoder-ffn-embed-dim $encoder_ffn_dim \
       --decoder-ffn-embed-dim $decoder_ffn_dim \
       --encoder-embed-dim $emb_size \
       --decoder-embed-dim $emb_size \
       >> $model_dir/train.log 


