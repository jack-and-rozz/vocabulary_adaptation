#!/bin/bash
echo "Running '$0 $1 $2'..."
 
. ./const.sh 

#####################################
##       unk rates
#####################################
# langs=(en ja)
# src_domain=jesc_sp
# tgt_domain=aspec_sp
langs=(de en)
src_domain=opus_it_sp
tgt_domain=opus_acquis_sp
src_domain_wd=$(remove_tok_suffix $src_domain)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
para_mode=${src_domain}16000.outD.v_${tgt_domain}16000_100k.all 
para_spm_dir=$(get_data_dir $para_mode $tgt_domain)
mono_mode=${src_domain}16000@${tgt_domain}16000.va.v_${tgt_domain}16000_100kmono.llm-idt.nn10.100k 
mono_spm_dir=$(get_data_dir $mono_mode $tgt_domain)

src_data_dir_wd=$(get_data_dir $para_mode $src_domain_wd)
tgt_data_dir_wd=$(get_data_dir $para_mode $tgt_domain_wd)
para_src_data_dir=$(get_data_dir $para_mode $src_domain)
para_tgt_data_dir=$(get_data_dir $para_mode $tgt_domain)
mono_src_data_dir=$(get_data_dir $mono_mode $src_domain)
mono_tgt_data_dir=$(get_data_dir $mono_mode $tgt_domain)
# lang=en
# echo "$para_src_data_dir/train.$lang $para_tgt_data_dir/train.$lang.100k $para_tgt_data_dir/test.$lang "
# echo " --mono_data $mono_src_data_dir/train.$lang $mono_tgt_data_dir/train.$lang.100k $mono_tgt_data_dir/test.$lang"
# exit 1
for lang in ${langs[@]}; do
    echo "-----"$lang"-----"
    python scripts/check_mono_effects.py \
	   $para_spm_dir/dict.$lang.txt \
	   $mono_spm_dir/dict.$lang.txt \
	   --para_data $para_src_data_dir/train.all.$lang $para_tgt_data_dir/train.100k.$lang $para_tgt_data_dir/test.$lang \
	   --mono_data $mono_tgt_data_dir/train.100k.$lang $mono_tgt_data_dir/test.$lang \
	   --wd_data $src_data_dir_wd/train.$lang $tgt_data_dir_wd/train.100k.$lang $tgt_data_dir_wd/test.$lang 
done
exit 1


    # case $mode in
    # 	*.${outdomain_suffix}.${size})
    # 	    data_dir=$base_data_dir.v_${tgt_domain}${tgt_vocab_size}.train-all
    # 	    ;;
    # 	*.${indomain_suffix}.${size})
    # 	    data_dir=$data_dir.v_${tgt_domain}${tgt_vocab_size}.train-${size}
    # 	    ;;
    # 	*.${outdomain_ext}.v_${tgt_domain}${tgt_vocab_size}.*)
    # 	    ;;
    # 	*.${direction_tok}*.${finetune_ext}.v_${src_domain}${src_vocab_size}.*)
    # 	    ;;
    # 	*.${direction_tok}*.${finetune_ext}.v_${tgt_domain}${tgt_vocab_size}.*)
    # 	    ;;
    # 	*.${direction_tok}*.${vocabadapt_ext}.v_${tgt_domain}.*)
    # 	    ;;
    # 	*.${direction_tok}*.${vocabadapt_ext}+mono.v_${tgt_domain}.*)
    # 	    ;;

    # 	*.${multidomain_ext}.*)
    # 	    echo "use a function get_multidomain_data_dir() instead."
    # 	    exit 1
    # 	    ;;
    # 	*.${backtranslation_ext}.*)
    # 	    echo "use a function get_multidomain_data_dir() instead."
    # 	    exit 1
    # 	    ;;
    # 	* ) echo "invalid mode: $mode"
    # 	    exit 1 ;;
    # esac



