
usage() {
    echo "Usage:$0 tgt_domain"
    exit 1
}

# if [ $# -lt 1 ];then
#     usage;
# fi
. ./const.sh

tgt_domain=$1

if [ ! -z $tgt_domain ]; then
    trained_models=$(ls $ckpt_root | grep 2${tgt_domain})
    trained_models=($tgt_domain.baseline ${trained_models[@]}) 
    for size in ${finetune_sizes[@]}; do
	if [ $size != all ]; then
	    trained_models=($tgt_domain.baseline.$size ${trained_models[@]}) 
	fi
    done
else
    trained_models=$(ls $ckpt_root)
fi

for model in ${trained_models[@]}; do
    model=$ckpt_root/$model
    if [ ! -e $model/train.log ]; then
	continue
    fi
    best_loss=$(cat $model/train.log | grep best_loss | tail -n 1 | cut -d '|' -f8- | cut -d ' ' -f3)

    is_done=$(tail -n1 $model/train.log | grep done)
    #echo $is_done

    if [ ! -z "$is_done" ]; then
	is_done="(DONE)"
    fi

    echo $model $best_loss $is_done

done
