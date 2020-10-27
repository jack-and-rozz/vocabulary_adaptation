######################################
###       Util bash functions
######################################

validate_mode(){
    mode=$1
    task=$2

    # if [[ ${mode} =~ ([0-9a-zA-Z_\-]+)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
    # 	src_domain=${BASH_REMATCH[1]}
    # 	tgt_domain=${BASH_REMATCH[2]}
    # elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
    # 	src_domain=${BASH_REMATCH[1]}
    # 	tgt_domain=${BASH_REMATCH[2]}
    # elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+)\.$outdomain_ext ]]; then
    # 	src_domain=${BASH_REMATCH[1]}
    # 	tgt_domain=${BASH_REMATCH[1]}
    # elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+)\.$indomain_ext ]]; then
    # 	src_domain=${BASH_REMATCH[1]}
    # 	tgt_domain=${BASH_REMATCH[1]}
    # else
    # 	echo "Invalid mode: $mode"
    # fi

    if [ $task == translation ]; then
	:
    else
	echo "Invalid task: $task"
	exit 1
    fi

}

parse_src_domain(){
    mode=$1
    if [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$outdomain_ext ]]; then
	src_domain=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$indomain_ext ]]; then
	src_domain=${BASH_REMATCH[1]}
    fi

    if [[ $src_domain =~ (.+_${sp_suffix}) ]]; then
	src_domain=${BASH_REMATCH[1]}
    fi
    echo $src_domain
}

parse_tgt_domain(){
    mode=$1
    if [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)${direction_tok}([0-9a-zA-Z_\-]+)\. ]]; then
	src_domain=${BASH_REMATCH[1]}
	tgt_domain=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$outdomain_ext.v_(.+?)\. ]]; then
    	tgt_domain=${BASH_REMATCH[2]}

    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$outdomain_ext ]]; then
	tgt_domain=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ ([0-9a-zA-Z_\-]+?)\.$indomain_ext ]]; then
	tgt_domain=${BASH_REMATCH[1]}
    fi

    if [[ $tgt_domain =~ (.+_${sp_suffix}) ]]; then
	tgt_domain=${BASH_REMATCH[1]}
    fi
    echo $tgt_domain

}

remove_tok_suffix(){
    domain=$1
    if [[ $domain =~ _$sp_suffix ]]; then
	l=$(expr $(expr length $sp_suffix) + 1) # remove '_sp', '_uni' or '_bpe'.
	domain=${domain:0:-$l}
    fi
    echo $domain
}

parse_emb_type(){
    mode=$1
    if [[ ${mode} =~ \.([llm|linear].+?)\.(nn[0-9]+)\. ]]; then
	emb_type=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ \.([llm|linear].+?)\.(.+) ]]; then
	emb_type=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ \.${finetune_ext}\.(.+?)\.(.+) ]]; then
	emb_type=${BASH_REMATCH[1]}
    fi

    echo $emb_type
}
parse_multidomain_type(){
    mode=$1
    if [[ ${mode} =~ \.${multidomain_ext}\.(.+?)\. ]]; then
	multidomain_type=${BASH_REMATCH[1]}
    fi
    echo $multidomain_type
}

parse_size(){
    mode=$1
    if [[ ${mode} =~ \.([0-9]+k) ]]; then
	size=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ .all ]]; then
	size=all
    fi
    if [ -z $size ]; then
	echo all
    else
	echo $size
    fi
}

parse_llm_nn(){
    mode=$1
    if [[ ${mode} =~ \.llm.+\.nn([0-9]+)\.* ]]; then
	num_nn=${BASH_REMATCH[1]}
    fi
    echo $num_nn
}


parse_src_vocab_size(){
    mode=$1
    if [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)${direction_tok} ]]; then
	vocab_size=${BASH_REMATCH[1]}
    elif [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[1]}
    fi
    echo $vocab_size
}

parse_tgt_vocab_size(){
    mode=$1
    if [[ ${mode} =~ ${direction_tok}(.+)[${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[2]}
    elif [[ ${mode} =~ [${word_suffix}|${sp_suffix}|${bpe_suffix}]([0-9]+)\. ]]; then
	vocab_size=${BASH_REMATCH[1]}
    fi
    echo $vocab_size
}

parse_fixed(){
    mode=$1
    if [[ ${mode} =~ \.($fixed_emb_suffix) ]]; then
	fixed=${BASH_REMATCH[1]}
    fi
    echo $fixed
}

get_src_lang(){
    domain=$1
    task=$2

    if [ $task != translation ]; then
	echo src
    else
	# lang=echo $(eval echo '$'$domain'_src_lang')
	# if [ ! -z $lang ]; then
	#     domain_wd=$(remove_tok_suffix $domain)
	#     lang=$(eval echo '$'$domain_wd'_src_lang')
	# fi
	if [[  $domain =~ _${sp_suffix} ]]; then
	    domain_wd=$(remove_tok_suffix $domain)
	    echo $(eval echo '$'$domain_wd'_src_lang')
	else
	    echo $(eval echo '$'$domain'_src_lang')
	fi
	echo $lang
    fi
}
get_tgt_lang(){
    domain=$1
    task=$2
    if [ $task != translation ]; then
	echo tgt
    else
	if [[  $domain =~ _${sp_suffix} ]]; then
	    domain_wd=$(remove_tok_suffix $domain)
	    echo $(eval echo '$'$domain_wd'_tgt_lang')
	else
	    echo $(eval echo '$'$domain'_tgt_lang')
	fi
    fi
}

parse_spm_mono_size(){
    mode=$1
    side=$2

    src_domain=$(parse_src_domain $mode)
    tgt_domain=$(parse_tgt_domain $mode)
    src_vocab_size=$(parse_src_vocab_size $mode)
    tgt_vocab_size=$(parse_tgt_vocab_size $mode)
    train_size=$(parse_size $mode)
    # if [[ $mode =~ \.v_([a-zA-z]+_sp[0-9]+)_([0-9kalmon]+) ]]; then
    if [[ $mode =~ \.v_([a-zA-z]+_sp[0-9]+)_([0-9a-zA-Z]+) ]]; then
	tgt_spm_domain=${BASH_REMATCH[1]}
	tgt_spm_mono_size=${BASH_REMATCH[2]}
    else
	tgt_spm_domain=$tgt_domain$tgt_vocab_size
	tgt_spm_mono_size=$train_size
    fi

    if [[ $mode =~ \.${vocabadapt_ext}\. ]] || [[ $mode =~ \.${vocabadapt_ext}_enc\. ]] || [[ $mode =~ \.${vocabadapt_ext}_dec\. ]]; then
	src_spm_domain=$src_domain$src_vocab_size
	src_spm_mono_size=all

    elif [[ $mode =~ \.${backtranslation_ext}_${vocabadapt_ext} ]]; then
	src_spm_domain=$src_domain$src_vocab_size
	src_spm_mono_size=all

    else
	src_spm_domain=$tgt_spm_domain
	src_spm_mono_size=$tgt_spm_mono_size
    fi

    if [ $side == src ]; then
	echo $src_spm_mono_size
    elif [ $side == tgt ]; then
	echo $tgt_spm_mono_size
    else
	exit 1
    fi
}

parse_spm_domain(){
    mode=$1
    side=$2

    src_domain=$(parse_src_domain $mode)
    tgt_domain=$(parse_tgt_domain $mode)
    src_vocab_size=$(parse_src_vocab_size $mode)
    tgt_vocab_size=$(parse_tgt_vocab_size $mode)
    train_size=$(parse_size $mode)
    if [[ $mode =~ \.v_([a-zA-z]+_sp[0-9]+)_([0-9kalmon]+) ]]; then
	tgt_spm_domain=${BASH_REMATCH[1]}
	tgt_spm_mono_size=${BASH_REMATCH[2]}
    else
	tgt_spm_domain=$src_domain$src_vocab_size
	tgt_spm_mono_size=$train_size
    fi

    # if [[ $mode =~ \.${vocabadapt_ext}\. ]]; then
    if [[ $mode =~ \.${vocabadapt_ext}\. ]] || [[ $mode =~ \.${vocabadapt_ext}_enc\. ]] || [[ $mode =~ \.${vocabadapt_ext}_dec\. ]]; then

	src_spm_domain=$src_domain$src_vocab_size
	src_spm_mono_size=all
    elif [[ $mode =~ \.${backtranslation_ext}_${vocabadapt_ext}\. ]]; then
	src_spm_domain=$src_domain$src_vocab_size
	src_spm_mono_size=all
    else
	src_spm_domain=$tgt_spm_domain
	src_spm_mono_size=$tgt_spm_mono_size
    fi

    if [ $side == src ]; then
	echo $src_spm_domain
    elif [ $side == tgt ]; then
	echo $tgt_spm_domain
    else
	exit 1
    fi
}

get_data_dir(){
    mode=$1
    domain=$2 # (e.g., jesc_sp)
    src_domain=$(parse_src_domain $mode)
    tgt_domain=$(parse_tgt_domain $mode)
    src_vocab_size=$(parse_src_vocab_size $mode)
    tgt_vocab_size=$(parse_tgt_vocab_size $mode)

    size=$(parse_size $mode)

    # if [[ $mode =~ \.${multidomain_ext}\.domainweighting ]]; then
    # 	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainweighting)
    # 	echo $data_dir
    # elif [[ $mode =~ \.${multidomain_ext}\.domainmixing ]]; then
    # 	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain domainmixing)
    # 	echo $data_dir
    # 	exit 1
    # elif [[ $mode =~ \.(${backtranslation_ext}_[a-z]+)\. ]]; then
    # 	bt_type=${BASH_REMATCH[1]}
    # 	data_dir=$(get_multidomain_data_dir $mode $src_domain $tgt_domain $bt_type)
    # 	echo $data_dir
    # 	exit 1
    # fi

    if [[ ! $domain =~ _${sp_suffix} ]]; then
	base_data_dir=$(eval echo '$'${domain}'_data_dir')
	if [ -z $base_data_dir ]; then
	    exit 1
	fi
	echo $base_data_dir
	exit 1
    else
	domain_wd=$(remove_tok_suffix $domain)
	base_data_dir=$(eval echo '$'${domain_wd}'_data_dir')

    fi

    # monolingual_type=$(parse_monolingual_type $mode)
    src_spm_domain=$(parse_spm_domain $mode src)
    tgt_spm_domain=$(parse_spm_domain $mode tgt)
    src_spm_mono_size=$(parse_spm_mono_size $mode src)
    tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

    if [[ $mode =~ (.+)${direction_tok}(.+).${vocabadapt_ext} ]]; then
    	# jesc_sp16000@aspec_sp16000k.va.v_aspec_sp16000_all.linear-idt.nn10.all
    	if [ $domain == $src_domain ]; then
    	    data_dir=$base_data_dir/$sp_suffix/v_${src_spm_domain}_${src_spm_mono_size}
    	elif [ $domain == $tgt_domain ]; then
    	    data_dir=$base_data_dir/$sp_suffix/v_${tgt_spm_domain}_${tgt_spm_mono_size}
    	fi
    elif [[ $mode =~ \.$multidomain_ext\. ]]; then
	echo "Invalid mode: $mode"
	exit 1
    elif [[ $mode =~ \.$backtranslation_ext\. ]]; then
	echo "Invalid mode: $mode"
	exit 1
    else 
	data_dir=$base_data_dir/$sp_suffix/v_${src_spm_domain}_${src_spm_mono_size}
    fi
    echo $data_dir
}

get_model_dir(){
    ckpt_root=$1
    mode=$2
    src_domain=$(parse_src_domain $mode)
    src_vocab_size=$(parse_src_vocab_size $mode)
    fixed=$(parse_fixed $mode)

    if [[ $mode =~ $sp_suffix ]]; then
	src_domain=$src_domain$src_vocab_size
    fi

    case $mode in
	*${direction_tok}*.noadapt*)
	    # Evaluate the source domain model in the target domain.
	    model_dir=$ckpt_root/${src_domain}.${outdomain_ext}.all
	    if [ ! -z $fixed ]; then
		model_dir=$model_dir.fixed
	    fi
	    ;;

	# *${direction_tok}*.${backtranslation_ext}_aug*)
	#     # Share the model trained in the same source domain.
	#     model_dir=$ckpt_root/${src_domain}${src_vocab_size}.backtranslation_aug
	#     if [ ! -z $fixed ]; then
	# 	model_dir=$model_dir.fixed
	#     fi
	#     ;;
	*)
	    model_dir=$ckpt_root/$mode
	;;
    esac
    echo $model_dir
}


# Get a path to the dataset constructed from a pair of domains.
get_multidomain_data_dir(){
    mode=$1
    src_domain=$2
    tgt_domain=$3
    mdtype=$4
    data_size=$(parse_size $mode)

    tgt_spm_domain=$(parse_spm_domain $mode tgt)
    tgt_spm_mono_size=$(parse_spm_mono_size $mode tgt)

    if [[ $src_domain =~ _${sp_suffix} ]] && [[ $tgt_domain =~ _${sp_suffix} ]]; then
	src_vocab_size=$(parse_src_vocab_size $mode)
	tgt_vocab_size=$(parse_tgt_vocab_size $mode)

	# The vocabulary size of a backward model is defined in source domain.
	if [[ $mdtype =~ ${backtranslation_ext}_aug ]]; then
	    vocab_size=$src_vocab_size
	# elif [[ $mdtype =~ ${backtranslation_ext}_ft ]] || [[ $mdtype =~ ${backtranslation_ext}_va ]]; then
	#     vocab_size=$tgt_vocab_size
	else
	    vocab_size=$tgt_vocab_size
	fi
	src_domain_wd=$(remove_tok_suffix $src_domain)
	tgt_domain_wd=$(remove_tok_suffix $tgt_domain)
	data_dir=$(eval echo '$'${src_domain_wd}'2'${tgt_domain_wd}'_data_dir')/$mdtype
	if [[ $mdtype =~ ${backtranslation_ext}_aug ]]; then
	    # TODO: fix format 
	    data_dir=$data_dir.${sp_suffix}${vocab_size}_${data_size}
	else
	    data_dir=$data_dir.v_${tgt_spm_domain}_${tgt_spm_mono_size}.${data_size}
	fi
    elif [[ ! $src_domain =~ _${sp_suffix} ]] && [[ ! $tgt_domain =~ _${sp_suffix} ]]; then
	data_dir=$(eval echo '$'$src_domain'2'$tgt_domain'_data_dir')/$mdtype
    fi
    echo $data_dir
}

get_backtranslation_type(){
    mode=$1
    if [[ $mode =~ .(backtranslation.+)\.(.+) ]]; then
	bt_type=${BASH_REMATCH[1]}
    elif [[ $mode =~ .(backtranslation.+)\.? ]]; then
	bt_type=${BASH_REMATCH[1]}
    fi
    echo $bt_type
}


get_domain_token(){
    domain=$1
    if [[ $domain =~ _${sp_suffix} ]]; then
	domain_wd=$(remove_tok_suffix $domain)
	domain_token=$(eval echo '$'$domain_wd'_domain_token')
	if [ -z $domain_token ]; then
	    echo '$'$(remove_tok_suffix $domain)'_domain_token' is not defined!
	    exit 1
	fi

    else
	domain_token=$(eval echo '$'$domain'_domain_token')
	if [ -z $domain_token ]; then
	    echo '$'${domain}'_domain_token' is not defined!
	    exit 1
	fi
    fi
    echo $domain_token
}

