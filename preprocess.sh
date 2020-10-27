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
is_valid=$(validate_mode $mode $task)
if [ -n "$is_valid" ]; then
    exit 1
fi
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
size=$(parse_size $mode)
src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

suffix=".$size"

src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)

case $mode in
    ########################################
    ###          Out/In-domain
    ########################################

    *.${outdomain_ext}.v_${tgt_domain}*)
	data_dir=$src_data_dir
	train_options="--max-update $train_steps"
	task_options="--task ${fairseq_task} \
		      --source-lang ${src_lang} \
		      --target-lang ${tgt_lang}
		     "
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *.${outdomain_ext}.*)
	data_dir=$src_data_dir
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data     \
		 --validpref $dev_data   \
		 --testpref $test_data 
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *.${indomain_ext}.*)
	data_dir=$tgt_data_dir
	data=$data_dir/fairseq.$size

	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data     \
		 --validpref $dev_data   \
		 --testpref $test_data 
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;


    *${direction_tok}*.noadapt*)
	data_dir=$tgt_data_dir
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    ########################################
    ###          Fine-tuning
    ########################################

    *${direction_tok}*.${finetune_ext}.*)
	data_dir=$tgt_data_dir
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.${vocabadapt_ext}.*)
	data_dir=$tgt_data_dir
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.${vocabadapt_ext}_enc.*)
	data_dir=$tgt_data_dir.enc
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.${vocabadapt_ext}_dec.*)
	data_dir=$tgt_data_dir.dec
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;


    # ########################################
    # ###       Back-translation
    # ########################################

    # Preprocess the dataset to train a model for data augmentation.
    *${direction_tok}*.${backtranslation_ext}_aug.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
					    ${backtranslation_ext}_aug)

	train_data=$data_dir/train.all
	dev_data=$data_dir/dev
	test_data=$data_dir/mono.ft${size}

    	# Swap src-lang and tgt-lang for bt.
    	options="--source-lang ${tgt_lang} \
	         --target-lang ${src_lang} \
    		 --trainpref $train_data  \
    		 --validpref $dev_data    \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${tgt_lang}.txt
    	tgt_dict=$data_dir/dict.${src_lang}.txt
	;;

    *${direction_tok}*.${backtranslation_ext}_ft.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
		   ${backtranslation_ext}_ft)

	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

    	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
    		 --trainpref $train_data \
    		 --validpref $dev_data \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${src_lang}.txt
    	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;
    *${direction_tok}*.${backtranslation_ext}_va.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain \
                   ${backtranslation_ext}_va)

	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev
	test_data=$data_dir/test

    	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
    		 --trainpref $train_data \
    		 --validpref $dev_data \
    		 --testpref $test_data"
    	src_dict=$data_dir/dict.${src_lang}.txt
    	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;



    ########################################
    ###     Multidomain-learning
    ########################################

    # Train with all of JESC dataset + part of ASPEC dataset.
    *${direction_tok}*.${multidomain_ext}.domainweighting.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
	data=$data_dir/fairseq.$size
	tgt_domain=$(remove_tok_suffix $tgt_domain)
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev.${tgt_domain}
	test_data=$data_dir/test.${tgt_domain}

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data \
		 --extra-features domain
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    *${direction_tok}*.${multidomain_ext}.domainmixing.*)
	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
	data=$data_dir/fairseq.$size
	tgt_domain=$(remove_tok_suffix $tgt_domain)
	train_data=$data_dir/train.$size
	dev_data=$data_dir/dev.${tgt_domain}
	test_data=$data_dir/test.${tgt_domain}

	options="--source-lang ${src_lang} --target-lang ${tgt_lang} \
		 --trainpref $train_data \
		 --validpref $dev_data \
		 --testpref $test_data
		 "
	src_dict=$data_dir/dict.${src_lang}.txt
	tgt_dict=$data_dir/dict.${tgt_lang}.txt
	;;

    * ) echo "invalid mode: $mode"
        exit 1
	;;
esac



if [ ! -z $size ] && [ $size != all ] && [ ! -e $data_dir/train.$size.${src_lang} ]; then
    python scripts/random_pickup.py \
	   $data_dir/train.${src_lang} \
	   $data_dir/train.${tgt_lang} $size
fi

destdir=$data_dir/fairseq$suffix

if [ $task != translation ]; then
    options="$options --srcdict $src_dict  --joined-dictionary"
else
    options="$options --srcdict $src_dict --tgtdict $tgt_dict"
fi

if [ ! -e $destdir ] || [ ! -n "$(ls $destdir)" ]; then

    if [ $size == all ] && [ -e $data_dir/fairseq ] && [ -z $suffix ]; then
	ln -sf fairseq $data_dir/fairseq.all
    else
	echo "Creating binary files with fairseq format to '$destdir'..."

	mkdir -p $destdir
	python fairseq/preprocess.py \
	       --destdir $destdir \
	       $options \
	       --workers 8
    fi

fi



