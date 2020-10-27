#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 model_dir mode task"
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


shard_id=0 # Test file ID for generation. The correspondense is defined by args.testpref in preprocess.sh.
num_shards=1 # Number of testing files.

size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 
input_lang=$src_lang
output_lang=$tgt_lang


model_dir=$(get_model_dir $ckpt_root $mode)
data_dir=$(get_data_dir $mode $tgt_domain)

if [[ $mode =~ $outdomain_ext ]]; then
    tgt_domain=$src_domain
fi

if [ ! -e $model_dir/tests ];then
    mkdir $model_dir/tests
fi

case $mode in
    #####################################
    ##        Translation
    #####################################

    ##### Test in ASPEC #####
    *.${outdomain_ext}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	
	data=$data_dir/fairseq.$size
	;;
    *.${indomain_ext}*)
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    ## w/o fine-tuning
    *${direction_tok}*.noadapt*) # Test ASPEC data with JESC vocabulary, by JESC model
	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	# ./preprocess.sh $mode $task
	;;

    ## w/ fine-tuning
    *${direction_tok}*.${finetune_ext}.*)
 	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    ## w/ fine-tuning + vocab adaptation
    *${direction_tok}*.${vocabadapt_ext}.*)
 	data_dir=$(get_data_dir $mode $tgt_domain)
	data=$data_dir/fairseq.$size
	;;

    *${direction_tok}*.${vocabadapt_ext}_enc.*)
 	data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir=$data_dir.enc
	data=$data_dir/fairseq.$size
	;;
    *${direction_tok}*.${vocabadapt_ext}_dec.*)
 	data_dir=$(get_data_dir $mode $tgt_domain)
	data_dir=$data_dir.dec
	data=$data_dir/fairseq.$size
	;;


    ## Multi-domain
    *${direction_tok}*.${multidomain_ext}.domainmixing.*) 
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	data=$data_dir/fairseq.$size
	;;
    *${direction_tok}*.${multidomain_ext}.domainweighting.*) 
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
	data=$data_dir/fairseq.$size
	;;

    ## Back-translation
    *${direction_tok}*.${backtranslation_ext}_aug.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_aug)
	data=$data_dir/fairseq.$size
	output_lang=$src_lang
	input_lang=$tgt_lang
	data_options="--skip-invalid-size-inputs-valid-test"
	;;

    *${direction_tok}*.${backtranslation_ext}_ft.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_ft)
	data=$data_dir/fairseq.$size
	;;
    *${direction_tok}*.${backtranslation_ext}_va.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain ${backtranslation_ext}_va)
	data=$data_dir/fairseq.$size
	;;
    * ) echo "invalid mode: $mode"
        exit 1 
	;;
esac
data_options="$data $data_options"

ckpt_options="--path $model_dir/checkpoints/checkpoint_best.pt \
	      --results-path $model_dir/tests/$tgt_domain "

if [ ! -e $model_dir/checkpoints/checkpoint_best.pt ]; then
    echo "$model_dir/checkpoints/checkpoint_best.pt was not found."
    exit 1
fi

# If the output file is older than the best checkpoint
log_file=$model_dir/tests/$tgt_domain.log
output_file=$model_dir/tests/$tgt_domain.outputs
input_file=$model_dir/tests/$tgt_domain.inputs


# Run testing again if the outputs were older than the best checkpoint.
test $model_dir/tests/$tgt_domain.outputs -nt $model_dir/checkpoints/checkpoint_best.pt
subword_output_is_latest=$?

if [ -e $model_dir/tests/$tgt_domain.outputs ] && [ $subword_output_is_latest == 0 ]; then
    echo "$model_dir/tests/$tgt_domain.outputs is up-to-date."
    exit 1
else
    echo "Evaluating $model_dir..."
    python fairseq/generate.py \
    	   --user-dir ${fairseq_user_dir} \
    	   --beam ${beam_size} \
    	   --lenpen $length_penalty \
    	   --task ${fairseq_task} \
    	   --shard-id $shard_id \
    	   --num-shards $num_shards \
    	   $ckpt_options \
    	   $data_options \
    	   > $log_file
fi

if [ ! -s $log_file ]; then
    echo "Error: $log_file is an empty file."
    exit 1
fi

if [[ ${mode} =~ domainmixing ]]; then
    # Remove the first generated token when domain mixing is employed.
    cat $model_dir/tests/$tgt_domain.log | grep "^H-" | cut -f1,3 | cut -c 3- | sort -k 1 -n | cut -f2 | cut -d ' ' -f2- > $model_dir/tests/$tgt_domain.outputs
else
    cat $model_dir/tests/$tgt_domain.log | grep "^H-" | cut -f1,3 | cut -c 3- | sort -k 1 -n | cut -f2 > $model_dir/tests/$tgt_domain.outputs
fi

if [ ! -e $model_dir/tests/$tgt_domain.inputs ]; then
    cat $model_dir/tests/$tgt_domain.log | grep "^S-" | cut -c 3- |sort -k 1 -n | cut -f2 > $model_dir/tests/$tgt_domain.inputs
fi
if [ ! -e $model_dir/tests/$tgt_domain.refs ]; then
    cat $model_dir/tests/$tgt_domain.log | grep "^T-" | cut -c 3- |sort -k 1 -n | cut -f2 > $model_dir/tests/$tgt_domain.refs
fi
 
if [[ ${mode} =~ _${sp_suffix} ]]; then
    tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
    word_level_outputs=$model_dir/tests/${tgt_domain_wd}.outputs
    word_level_refs=$model_dir/tests/${tgt_domain_wd}.refs
    word_level_inputs=$model_dir/tests/${tgt_domain_wd}.inputs

    # Run decoding again if the word-level outputs were older than the subword-level outputs.
    test $word_level_outputs -nt $output_file
    word_output_is_latest=$?
    if [ $word_output_is_latest != 0 ]; then
	echo "Applying subword detokenization to $model_dir/tests/$tgt_domain.outputs..."
	if [ -e $model_dir/subword/spm.$output_lang.model ]; then
	    spm_decode --model $model_dir/subword/spm.$output_lang.model \
		       --output $word_level_outputs \
		       < $model_dir/tests/$tgt_domain.outputs
	    spm_decode --model $model_dir/subword/spm.$output_lang.model \
		       --output $word_level_refs \
		       < $model_dir/tests/$tgt_domain.refs
	    spm_decode --model $model_dir/subword/spm.$input_lang.model \
		       --output $word_level_inputs \
		       < $model_dir/tests/$tgt_domain.inputs
	else
	    spm_decode --model $data_dir/spm.$output_lang.model \
		       --output $word_level_outputs \
		       < $model_dir/tests/$tgt_domain.outputs
	    spm_decode --model $data_dir/spm.$output_lang.model \
		       --output $word_level_refs \
		       < $model_dir/tests/$tgt_domain.refs
	    spm_decode --model $data_dir/spm.$input_lang.model \
		       --output $word_level_inputs \
		       < $model_dir/tests/$tgt_domain.inputs

	fi
    fi
fi
