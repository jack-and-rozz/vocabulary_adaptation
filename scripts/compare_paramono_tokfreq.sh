#!/bin/bash
echo "Running '$0 $1 $2'..."
 
. ./const.sh 

langs=(en ja)
src_domain=jesc_sp
tgt_domain=aspec_sp

# langs=(de en)
# src_domain=opus_it_sp
# tgt_domain=opus_acquis_sp

src_domain_wd=$(remove_tok_suffix $src_domain)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
para_mode=${src_domain}16000@${tgt_domain}16000.${finetune_ext}.v_${tgt_domain}16000_100k.100k
para_src_spm_dir=$(get_data_dir $para_mode $src_domain)
para_tgt_spm_dir=$(get_data_dir $para_mode $tgt_domain)
mono_mode=${src_domain}16000@${tgt_domain}16000.${finetune_ext}.v_${tgt_domain}16000_100kmono.100k
# mono_mode=${src_domain}16000@${tgt_domain}16000.${vocabadapt_ext}.v_${tgt_domain}16000_100kmono.llm-idt.nn10.100k 
mono_src_spm_dir=$(get_data_dir $mono_mode $src_domain)
mono_tgt_spm_dir=$(get_data_dir $mono_mode $tgt_domain)

src_data_dir_wd=$(get_data_dir $para_mode $src_domain_wd)
tgt_data_dir_wd=$(get_data_dir $para_mode $tgt_domain_wd)
para_src_data_dir=$(get_data_dir $para_mode $src_domain)
para_tgt_data_dir=$(get_data_dir $para_mode $tgt_domain)
mono_src_data_dir=$(get_data_dir $mono_mode $src_domain)
mono_tgt_data_dir=$(get_data_dir $mono_mode $tgt_domain)

# echo $para_src_spm_dir
# echo $para_tgt_spm_dir
# echo $mono_src_spm_dir
# echo $mono_tgt_spm_dir
# echo $para_src_data_dir
# echo $para_tgt_data_dir
# echo $mono_src_data_dir
# echo $mono_tgt_data_dir


echo "En-Ja"
for lang in ${langs[@]}; do
    echo "-----"$lang"-----"
    python scripts/compare_paramono_tokfreq.py \
	   $para_src_data_dir \
	   $para_tgt_data_dir \
	   $mono_src_data_dir \
	   $mono_tgt_data_dir \
	   $lang
done


exit 1

# langs=(en ja)
# src_domain=jesc_sp
# tgt_domain=aspec_sp

langs=(de en)
src_domain=opus_it_sp
tgt_domain=opus_acquis_sp

src_domain_wd=$(remove_tok_suffix $src_domain)
tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
para_mode=${src_domain}16000@${tgt_domain}16000.${finetune_ext}.v_${tgt_domain}16000_100k.100k
para_src_spm_dir=$(get_data_dir $para_mode $src_domain)
para_tgt_spm_dir=$(get_data_dir $para_mode $tgt_domain)

mono_mode=${src_domain}16000@${tgt_domain}16000.${finetune_ext}.v_${tgt_domain}16000_100kmono.100k
# mono_mode=${src_domain}16000@${tgt_domain}16000.${vocabadapt_ext}.v_${tgt_domain}16000_100kmono.llm-idt.nn10.100k 
mono_src_spm_dir=$(get_data_dir $mono_mode $src_domain)
mono_tgt_spm_dir=$(get_data_dir $mono_mode $tgt_domain)

src_data_dir_wd=$(get_data_dir $para_mode $src_domain_wd)
tgt_data_dir_wd=$(get_data_dir $para_mode $tgt_domain_wd)
para_src_data_dir=$(get_data_dir $para_mode $src_domain)
para_tgt_data_dir=$(get_data_dir $para_mode $tgt_domain)
mono_src_data_dir=$(get_data_dir $mono_mode $src_domain)
mono_tgt_data_dir=$(get_data_dir $mono_mode $tgt_domain)

# echo $para_src_data_dir
# echo $para_tgt_data_dir
# echo $mono_src_data_dir
# echo $mono_tgt_data_dir

echo "De-En"
for lang in ${langs[@]}; do
    echo "-----"$lang"-----"
    python scripts/compare_paramono_tokfreq.py \
	   $para_src_data_dir \
	   $para_tgt_data_dir \
	   $mono_src_data_dir \
	   $mono_tgt_data_dir \
	   $lang
done
exit 1
