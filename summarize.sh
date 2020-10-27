#!/bin/bash
echo "Running '$0 $@'..."
usage() {
    echo "Usage:$0 src_domain tgt_domain task (n_vocab)"
    exit 1
}
if [ $# -lt 3 ];then
    usage;
fi

. ./const.sh $mode $task
# mode=$1
# task=$2

src_domain=$1
tgt_domain=$2
task=$3
n_vocab=$4

if [ -z $n_vocab ]; then
    n_vocab=0
fi

src_domain=$(remove_tok_suffix $src_domain)
tgt_domain=$(remove_tok_suffix $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 

# (TODO)
src_data_dir=$(get_data_dir $src_domain.$indomain_ext.all $src_domain)
tgt_data_dir=$(get_data_dir $tgt_domain.$indomain_ext.all $tgt_domain)

input_file=$tgt_data_dir/test.$src_lang
reference_file=$tgt_data_dir/test.$tgt_lang
src_enc_vocab=$src_data_dir/dict.$src_lang.txt
src_dec_vocab=$src_data_dir/dict.$tgt_lang.txt
tgt_enc_vocab=$tgt_data_dir/dict.$src_lang.txt
tgt_dec_vocab=$tgt_data_dir/dict.$tgt_lang.txt
output_filename=$tgt_domain.outputs


# Apply normalization to the outputs, remove all spaces, and retokenize them when the output language is Japanese.
if [ $tgt_lang == ja ]; then
    if [ ! -e $reference_file.normed ]; then
	python -c "import unicodedata; [print(unicodedata.normalize('NFKC', s).strip()) for s in open('$reference_file')]" > $reference_file.normed
    fi
    for output_file in $(ls $ckpt_root/*/tests/$output_filename); do
	if [[ $output_file =~ \.${backtranslation_ext}_aug\. ]]; then
	    continue
	fi
	test $output_file.normed -ot $output_file
	if [ $? == 0 ]; then
	    echo $output_file
	    python -c "import unicodedata; [print(unicodedata.normalize('NFKC', ''.join(s.strip().split()))) for s in open('$output_file')]" > $output_file.normed
	    kytea -notags \
		  < $output_file.normed \
		  > $output_file.normed.retok &
	fi
    done
    wait
    output_filename=$output_filename.normed.retok
    reference_file=$reference_file.normed
fi


if [ -n "$src_enc_vocab" ]; then
    options="$options --src_enc_vocab $src_enc_vocab --src_dec_vocab $src_dec_vocab"
fi
if [ -n "$tgt_enc_vocab" ]; then
    options="$options --tgt_enc_vocab $tgt_enc_vocab --tgt_dec_vocab $tgt_dec_vocab"
fi

python scripts/summarize_result.py $ckpt_root \
       $output_filename $input_file $reference_file \
       --task $task \
       $options \
       --n_vocab 0

if [ $task == dialogue ]; then
    python scripts/calc_dist.py $ckpt_root $output_filename $reference_file
fi

if [ $task == descgen ]; then
    word_list=$tgt_data_dir/test.word
fi
if [ $task == descgen ] || [ $task == dialogue ]; then
    if [ -z $word_list ]; then
	python scripts/calc_bleu_descgen.py $ckpt_root $output_filename $reference_file --sentence_bleu_path=$sentence_bleu_path
    else
	python scripts/calc_bleu_descgen.py $ckpt_root $output_filename $reference_file --word_list=$word_list --sentence_bleu_path=$sentence_bleu_path
    fi
fi

