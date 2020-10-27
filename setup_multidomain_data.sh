#!/bin/bash
echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task "
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

# *NOTE* Subword tokenization for multidomain datasets is done in this script, but not in ./setup_sentencepiece.sh. This is because multidomain datasets require to copy extra files (e.g., .domain files for DomainWeighting) when converting word-based files to subword-based ones, and tokenization must be trained again from the joined training set of the souce and target domains.

src_domain=$(parse_src_domain $mode)
tgt_domain=$(parse_tgt_domain $mode)
src_vocab_size=$(parse_src_vocab_size $mode)
tgt_vocab_size=$(parse_tgt_vocab_size $mode)
ft_size=$(parse_size $mode)

if [ $src_vocab_size != $tgt_vocab_size ]; then
    echo 'ERROR: In MDL, the specified vocab size is the same across the two domains.'
    exit 1
fi
n_sentencepiece=$tgt_vocab_size

# Setup word-based multidomain datasets first, and then train/encode them into subword-based ones.


if [[ $src_domain =~ _${sp_suffix} ]] || [[ $src_domain =~ _$bpe_suffix ]];then
    use_subword=true
fi
src_domain=$(remove_tok_suffix $src_domain)
tgt_domain=$(remove_tok_suffix $tgt_domain)


src_data_dir=$(get_data_dir $mode $src_domain)
tgt_data_dir=$(get_data_dir $mode $tgt_domain)

src_lang=$(get_src_lang $tgt_domain $task) 
tgt_lang=$(get_tgt_lang $tgt_domain $task) 
src_domain_token=$(get_domain_token $src_domain)
tgt_domain_token=$(get_domain_token $tgt_domain)
data_types=(train dev test test2)
langs=($src_lang $tgt_lang)

########################################################
#         Setup for all MDL methods
#######################################################

if [ ! -e $src_data_dir -o ! -e $tgt_data_dir ]; then
    echo "Error: This script requires processed data in $src_data_dir and $tgt_data_dir. Run ./scripts/dataset/$src_domain/setup_dataset.sh and ./scripts/dataset/$tgt_domain/setup_dataset.sh first."
    exit 1
fi


# Do random sampling to tgt domain dataset if each subset does not exist.
for size in ${finetune_sizes[@]}; do
    if [ ! -e $tgt_data_dir/train.$size.${src_lang} ]; then
	echo "Sampling $size examples from $tgt_data_dir/train.${src_lang} and $tgt_data_dir/train.${tgt_lang}..."
	python scripts/random_pickup.py \
	       $tgt_data_dir/train.${src_lang} $tgt_data_dir/train.${tgt_lang} $size --seed $random_seed
    fi
done


####################################################
#         Setup for domain weighting
####################################################

domainweighting_data_dir=$(get_multidomain_data_dir $mode \
			   $src_domain $tgt_domain domainweighting)

if [ -z $domainweighting_data_dir ]; then
    echo "Error: Define \$${src_domain}@${tgt_domain}_domainweighting_data_dir in const.sh."
    exit 1
fi
if [ ! -e $domainweighting_data_dir ]; then
    mkdir -p $domainweighting_data_dir
fi

if [ ! -e $domainweighting_data_dir/train.${src_lang} ]; then
    echo "Creating a mixed dataset to $domainweighting_data_dir..."

    for dtype in ${data_types[@]}; do
	for lang in ${langs[@]}; do
	    if [ ! -e $domainweighting_data_dir/$dtype.$lang ] && [ -e $(pwd)/$src_data_dir/$dtype.${lang} ]; then
		cat $(pwd)/$src_data_dir/$dtype.${lang} \
		    > $domainweighting_data_dir/$dtype.${lang} &
	    fi
	done
    done
    wait 
    cat $(pwd)/$src_data_dir/train.${src_lang} > $domainweighting_data_dir/train.${src_lang} &
    cat $(pwd)/$src_data_dir/train.${tgt_lang} > $domainweighting_data_dir/train.${tgt_lang} &

    cp $(pwd)/$src_data_dir/dev.${src_lang}  $domainweighting_data_dir/dev.${src_domain}.${src_lang} &
    cp $(pwd)/$src_data_dir/dev.${tgt_lang}  $domainweighting_data_dir/dev.${src_domain}.${tgt_lang} &
    cp $(pwd)/$src_data_dir/test.${src_lang} $domainweighting_data_dir/test.${src_domain}.${src_lang} &
    cp $(pwd)/$src_data_dir/test.${tgt_lang} $domainweighting_data_dir/test.${src_domain}.${tgt_lang} &

    wait 
    cat $(pwd)/$tgt_data_dir/train.${src_lang} >> $domainweighting_data_dir/train.${src_lang} &
    cat $(pwd)/$tgt_data_dir/train.${tgt_lang} >> $domainweighting_data_dir/train.${tgt_lang} &
    cp $(pwd)/$tgt_data_dir/dev.${src_lang} $domainweighting_data_dir/dev.${tgt_domain}.${src_lang} &
    cp $(pwd)/$tgt_data_dir/dev.${tgt_lang} $domainweighting_data_dir/dev.${tgt_domain}.${tgt_lang} &
    cp $(pwd)/$tgt_data_dir/test.${src_lang} $domainweighting_data_dir/test.${tgt_domain}.${src_lang} &
    cp $(pwd)/$tgt_data_dir/test.${tgt_lang} $domainweighting_data_dir/test.${tgt_domain}.${tgt_lang} &


    if [ -e $(pwd)/$src_data_dir/test2.${src_lang} ]; then
	cp $(pwd)/$src_data_dir/test2.${src_lang} $domainweighting_data_dir/test2.${src_domain}.${src_lang} &
	cp $(pwd)/$src_data_dir/test2.${tgt_lang} $domainweighting_data_dir/test2.${src_domain}.${tgt_lang} &
	wait 
    fi
    if [ -e $(pwd)/$tgt_data_dir/test2.${src_lang} ]; then
	cp $(pwd)/$tgt_data_dir/test2.${src_lang} \
	   $domainweighting_data_dir/test2.${tgt_domain}.${src_lang} &
	cp $(pwd)/$tgt_data_dir/test2.${tgt_lang} \
	   $domainweighting_data_dir/test2.${tgt_domain}.${tgt_lang} &
    fi
fi

# Create a joined training set by all the source domain training set and the target domain fine-tuning set.
for size in ${finetune_sizes[@]}; do
    # Ignore if the size of fine-tuning set is larger than the original dataset and the small subset was not created.
    if [ ! -e $tgt_data_dir/train.$size.${src_lang} ]; then
	continue
    fi

    if [ ! -e $domainweighting_data_dir/train.$size.${src_lang} ]; then
	cat $src_data_dir/train.${src_lang} > $domainweighting_data_dir/train.$size.${src_lang} &
	cat $src_data_dir/train.${tgt_lang} > $domainweighting_data_dir/train.$size.${tgt_lang} &
	wait
	cat $tgt_data_dir/train.$size.${src_lang} >> $domainweighting_data_dir/train.$size.${src_lang} &
	cat $tgt_data_dir/train.$size.${tgt_lang} >> $domainweighting_data_dir/train.$size.${tgt_lang} &

    fi
done
wait

# Create a file of domain tokens
if [ ! -e  $domainweighting_data_dir/train.domain ]; then
    python -c "print('\n'.join(['$src_domain_token' for l in open('${src_data_dir}/train.${src_lang}')]))" > $domainweighting_data_dir/train.domain
    python -c "print('\n'.join(['$tgt_domain_token' for l in open('${tgt_data_dir}/train.${src_lang}')]))" >> $domainweighting_data_dir/train.domain
fi 

for size in ${finetune_sizes[@]}; do
    # Ignore if the size of fine-tuning set is larger than the original dataset and the small subset was not created.
    if [ ! -e $tgt_data_dir/train.$size.${src_lang} ]; then
	continue
    fi

    if [ ! -e  $domainweighting_data_dir/train.$size.domain ]; then
	python -c "print('\n'.join(['$src_domain_token' for l in open('${src_data_dir}/train.${src_lang}')]))" > $domainweighting_data_dir/train.$size.domain
	python -c "print('\n'.join(['$tgt_domain_token' for l in open('${tgt_data_dir}/train.${size}.${src_lang}')]))" >> $domainweighting_data_dir/train.$size.domain
    fi
done 

if [ ! -e $domainweighting_data_dir/dict.domain.txt ]; then
    echo "$src_domain_token" > $domainweighting_data_dir/dict.domain.txt
    echo "$tgt_domain_token" >> $domainweighting_data_dir/dict.domain.txt
fi


if [ ! -e $domainweighting_data_dir/dev.${src_domain}.domain ]; then
    python -c "print('\n'.join(['$src_domain_token' for l in open('${src_data_dir}/dev.${src_lang}')]))" > $domainweighting_data_dir/dev.${src_domain}.domain
fi

if [ ! -e $domainweighting_data_dir/dev.${tgt_domain}.domain ]; then
    python -c "print('\n'.join(['$tgt_domain_token' for l in open('${tgt_data_dir}/dev.${src_lang}')]))" > $domainweighting_data_dir/dev.${tgt_domain}.domain
fi

if [ ! -e $domainweighting_data_dir/test.${src_domain}.domain ]; then
    python -c "print('\n'.join(['$src_domain_token' for l in open('${src_data_dir}/test.${src_lang}')]))" > $domainweighting_data_dir/test.${src_domain}.domain
fi

if [ ! -e $domainweighting_data_dir/test.${tgt_domain}.domain ]; then
    python -c "print('\n'.join(['$tgt_domain_token' for l in open('${tgt_data_dir}/test.${src_lang}')]))" > $domainweighting_data_dir/test.${tgt_domain}.domain
fi

####################################################
#         Setup for domain mixing
####################################################

domainmixing_data_dir=$(get_multidomain_data_dir $mode \
			    $src_domain $tgt_domain domainmixing)

if [ -z $domainmixing_data_dir ]; then
    echo "Error: Define \$${src_domain}2${tgt_domain}_domainmixing_data_dir in const.sh."
    exit 1
fi
if [ ! -e $domainmixing_data_dir ]; then
    mkdir -p $domainmixing_data_dir
fi
if [ ! -e $domainmixing_data_dir/train.${src_lang} ]; then
    echo "Creating a mixed dataset to $domainmixing_data_dir..."
    cp $(pwd)/$domainweighting_data_dir/train.${src_lang} $domainmixing_data_dir/train.${src_lang} 
    cp $(pwd)/$domainweighting_data_dir/dev.${src_domain}.${src_lang} $domainmixing_data_dir/dev.${src_domain}.${src_lang} 
    cp $(pwd)/$domainweighting_data_dir/dev.${tgt_domain}.${src_lang} $domainmixing_data_dir/dev.${tgt_domain}.${src_lang} 
    cp $(pwd)/$domainweighting_data_dir/test.${src_domain}.${src_lang} $domainmixing_data_dir/test.${src_domain}.${src_lang}
    cp $(pwd)/$domainweighting_data_dir/test.${src_domain}.${tgt_lang} $domainmixing_data_dir/test.${src_domain}.${tgt_lang}

    cp $(pwd)/$domainweighting_data_dir/test.${tgt_domain}.${src_lang} $domainmixing_data_dir/test.${tgt_domain}.${src_lang}
    cp $(pwd)/$domainweighting_data_dir/test.${tgt_domain}.${tgt_lang} $domainmixing_data_dir/test.${tgt_domain}.${tgt_lang}
fi

# if [ -e $(pwd)/$domainweighting_data_dir/test2.${src_domain}.${src_lang} ]; then
#     cp $(pwd)/$domainweighting_data_dir/test2.${src_domain}.${src_lang} $domainmixing_data_dir/test2.${src_domain}.${src_lang}
#     cp $(pwd)/$domainweighting_data_dir/test2.${tgt_domain}.${src_lang} $domainmixing_data_dir/test2.${tgt_domain}.${src_lang}
# fi

# if [ -e $(pwd)/$domainweighting_data_dir/test2.${tgt_domain}.${src_lang} ]; then
#     cp $(pwd)/$domainweighting_data_dir/test2.${tgt_domain}.${src_lang} $domainmixing_data_dir/test2.${tgt_domain}.${src_lang}
#     cp $(pwd)/$domainweighting_data_dir/test2.${tgt_domain}.${tgt_lang} $domainmixing_data_dir/test2.${tgt_domain}.${tgt_lang}
# fi


# Use the same input as domainweighting.
for size in ${finetune_sizes[@]}; do
    # Ignore if the size of fine-tuning set is larger than the original dataset and the small subset was not created.
    if [ ! -e $tgt_data_dir/train.$size.${src_lang} ]; then
	continue
    fi

    if [ ! -e $domainmixing_data_dir/train.$size.${src_lang} ]; then
	cp $(pwd)/$domainweighting_data_dir/train.$size.${src_lang} $domainmixing_data_dir/train.$size.${src_lang} 
    fi
done

# Prepend a domain token to the target sentences of domainweighting.
if [ ! -e $domainmixing_data_dir/train.${tgt_lang} ]; then
    paste -d ' ' \
	  $domainweighting_data_dir/train.domain \
	  $domainweighting_data_dir/train.${tgt_lang} \
	  > $domainmixing_data_dir/train.${tgt_lang} &
fi

for size in ${finetune_sizes[@]}; do
    # Ignore if the size of fine-tuning set is larger than the original dataset and the small subset was not created.
    if [ ! -e $tgt_data_dir/train.$size.${src_lang} ]; then
	continue
    fi

    if [ ! -e $domainmixing_data_dir/train.$size.${tgt_lang} ]; then
	paste -d ' ' \
	      $domainweighting_data_dir/train.$size.domain \
	      $domainweighting_data_dir/train.$size.${tgt_lang} \
	      > $domainmixing_data_dir/train.$size.${tgt_lang} &
    fi
done
if [ ! -e $domainmixing_data_dir/dev.$size.${tgt_lang} ]; then
    paste -d ' ' \
	  $domainweighting_data_dir/dev.${src_domain}.domain \
	  $domainweighting_data_dir/dev.${src_domain}.${tgt_lang} \
	  > $domainmixing_data_dir/dev.${src_domain}.${tgt_lang} &
fi

if [ ! -e $domainmixing_data_dir/dev.$size.${tgt_lang} ]; then
    paste -d ' ' \
	  $domainweighting_data_dir/dev.${tgt_domain}.domain \
	  $domainweighting_data_dir/dev.${tgt_domain}.${tgt_lang} \
	  > $domainmixing_data_dir/dev.${tgt_domain}.${tgt_lang} &
fi


if [ $task != translation ] && [ ! -e $domainmixing_data_dir/train.flat ]; then
    cat $domainmixing_data_dir/train.${src_lang} > $domainmixing_data_dir/train.flat
    cat $domainmixing_data_dir/train.${tgt_lang} >> $domainmixing_data_dir/train.flat
fi
if  [ $task != translation ] && [ ! -e $domainweighting_data_dir/train.flat ]; then
    cat $domainweighting_data_dir/train.${src_lang} > $domainweighting_data_dir/train.flat
    cat $domainweighting_data_dir/train.${tgt_lang} >> $domainweighting_data_dir/train.flat
fi



##################################################
##      Monolingual corpus
##################################################

# if [ ! -e $domainweighting_data_dir/monolingual.$src_lang ]; then
#     ln -sf $(pwd)/$domainweighting_data_dir/train.$src_lang $domainweighting_data_dir/monolingual.$src_lang
#     ln -sf $(pwd)/$domainweighting_data_dir/train.$tgt_lang $domainweighting_data_dir/monolingual.$tgt_lang
#     ln -sf $(pwd)/$domainmixing_data_dir/train.$src_lang $domainmixing_data_dir/monolingual.$src_lang
#     ln -sf $(pwd)/$domainmixing_data_dir/train.$tgt_lang $domainmixing_data_dir/monolingual.$tgt_lang
# fi



##################################################
##    Training & Encoding of Sentencepiece 
##################################################

if [ ! -z $use_subword ]; then 
    domainweighting_sp_data_dir=$(get_multidomain_data_dir $mode ${src_domain}_${sp_suffix} ${tgt_domain}_${sp_suffix} domainweighting)
    domainmixing_sp_data_dir=$(get_multidomain_data_dir $mode ${src_domain}_${sp_suffix} ${tgt_domain}_${sp_suffix} domainmixing)

    if [ ! -e $domainweighting_sp_data_dir ]; then
	mkdir $domainweighting_sp_data_dir
    fi
    if [ ! -e $domainmixing_sp_data_dir ]; then
	mkdir $domainmixing_sp_data_dir
    fi

    # Train subword tokenization in each language.
    if [ ! -e $domainweighting_sp_data_dir/spm.$src_lang.model ]; then
	echo "training '$domainweighting_sp_data_dir/spm.$src_lang'..."
	spm_train --vocab_size ${n_sentencepiece} \
		  --model_prefix $domainweighting_sp_data_dir/spm.$src_lang \
		  --unk_surface $unk_surface \
		  --input_sentence_size 2000000 \
		  --shuffle_input_sentence \
		  --hard_vocab_limit=false \
		  --model_type=$spm_model_type \
		  --input $domainweighting_data_dir/train.$ft_size.$src_lang
    fi

    if [ ! -e $domainweighting_sp_data_dir/spm.$tgt_lang.model ]; then
	if [ $task == translation ]; then
	    echo "training '$domainweighting_sp_data_dir/spm.$tgt_lang'..."
	    spm_train --vocab_size ${n_sentencepiece} \
		      --model_prefix $domainweighting_sp_data_dir/spm.$tgt_lang \
		      --unk_surface $unk_surface \
		      --input_sentence_size 2000000 \
		      --shuffle_input_sentence \
		      --hard_vocab_limit=false \
		      --model_type=$spm_model_type \
		      --input $domainweighting_data_dir/train.$ft_size.$tgt_lang 
	else
	    ln -sf spm.$src_lang.model $domainweighting_sp_data_dir/spm.$tgt_lang.model 
	    ln -sf spm.$src_lang.vocab $domainweighting_sp_data_dir/spm.$tgt_lang.vocab

	fi
    fi

    if [ ! -e $domainmixing_sp_data_dir/spm.$src_lang.model ]; then
	echo "training '$domainmixing_sp_data_dir/spm.$src_lang'..."
	spm_train --vocab_size ${n_sentencepiece} \
		  --model_prefix $domainmixing_sp_data_dir/spm.$src_lang \
		  --unk_surface $unk_surface \
		  --input_sentence_size 2000000 \
		  --shuffle_input_sentence \
		  --hard_vocab_limit=false \
		  --model_type=$spm_model_type \
		  --input $domainmixing_data_dir/train.$ft_size.$src_lang \
		  --user_defined_symbols ▁$src_domain_token,▁$tgt_domain_token 
    fi
    if [ ! -e $domainmixing_sp_data_dir/spm.$tgt_lang.model ]; then
	if [ $task == translation ]; then
	    echo "training '$domainmixing_sp_data_dir/spm.$tgt_lang'..."
	    spm_train --vocab_size ${n_sentencepiece} \
		      --model_prefix $domainmixing_sp_data_dir/spm.$tgt_lang \
		      --unk_surface $unk_surface \
		      --input_sentence_size 2000000 \
		      --shuffle_input_sentence \
		      --hard_vocab_limit=false \
		      --model_type=$spm_model_type \
		      --input $domainmixing_data_dir/train.$ft_size.$tgt_lang \
		      --user_defined_symbols ▁$src_domain_token,▁$tgt_domain_token 
	else
	    ln -sf spm.$src_lang.model $domainmixing_sp_data_dir/spm.$tgt_lang.model 
	    ln -sf spm.$src_lang.vocab $domainmixing_sp_data_dir/spm.$tgt_lang.vocab

	fi
    fi
    wait

    langs=($src_lang $tgt_lang)
    files=(train.$ft_size dev.$src_domain dev.$tgt_domain test.$src_domain test.$tgt_domain)
    for lang in ${langs[@]}; do
    	# for input_path in $(ls $domainweighting_data_dir/*.$lang); do
	for file in ${files[@]}; do
	    # output_path=$domainweighting_sp_data_dir/$(basename $input_path)
	    output_path=$domainweighting_sp_data_dir/$file.$lang
	    input_path=$domainweighting_data_dir/$file.$lang
    	    if [ ! -e $output_path ]; then
    	    	echo "Encoding $input_path to $output_path by $domainweighting_sp_data_dir/spm.$lang.model..."
    	    	spm_encode --model $domainweighting_sp_data_dir/spm.$lang.model \
    	    		   --output $output_path \
    	    		   < $input_path &
    	    fi
    	done
    done
    # for path in $(ls $domainweighting_data_dir/*.domain); do
    for file in ${files[@]}; do
	if [ ! -e $domainweighting_sp_data_dir/$file.domain ]; then
	    cp $domainweighting_data_dir/$file.domain \
	       $domainweighting_sp_data_dir/$file.domain &
	fi
    done
    if [ ! -e $domainweighting_sp_data_dir/dict.domain.txt ]; then
	cp $domainweighting_data_dir/dict.domain.txt $domainweighting_sp_data_dir
    fi
    wait
    for lang in ${langs[@]}; do
    	# for input_path in  $(ls $domainmixing_data_dir/*.$lang); do
    	for file in ${files[@]}; do
	    # output_path=$domainmixing_sp_data_dir/$(basename $input_path)
	    output_path=$domainmixing_sp_data_dir/$file.$lang
	    input_path=$domainmixing_data_dir/$file.$lang
    	    if [ ! -e $output_path ]; then
    	    	echo "Encoding $input_path to $output_path by $domainmixing_sp_data_dir/spm.$lang.model..."
    	    	spm_encode --model $domainmixing_sp_data_dir/spm.$lang.model \
    	    		   --output $output_path \
    	    		   < $input_path &
    	    fi
    	done
    done
    wait
fi



if [ ! -e $domainweighting_sp_data_dir/monolingual.$src_lang ]; then
    ln -sf train.$ft_size.$src_lang $domainweighting_sp_data_dir/monolingual.$src_lang
    ln -sf train.$ft_size.$tgt_lang $domainweighting_sp_data_dir/monolingual.$tgt_lang
    ln -sf train.$ft_size.$src_lang $domainmixing_sp_data_dir/monolingual.$src_lang
    ln -sf train.$ft_size.$tgt_lang $domainmixing_sp_data_dir/monolingual.$tgt_lang
fi
