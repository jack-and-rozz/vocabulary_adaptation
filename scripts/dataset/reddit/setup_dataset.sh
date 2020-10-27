#!/bin/bash

. ./const.sh

original_dir=dataset/reddit/original
processed_dir=$reddit_data_dir

if [ ! -e $processed_dir ]; then
   mkdir -p $processed_dir
fi

# python scripts/reddit/setup_dataset.py -src  $original_dir/RC_2015-01

max_dialog_tokens=70
# python scripts/reddit/slice_dialogs.py \
#        $original_dir/train.${max_dialog_turns}turns.src \
#        $original_dir/train.${max_dialog_turns}turns.tgt \
#        --max-turns $max_dialog_turns \
#        --max-tokens $max_dialog_tokens \
#        < $original_dir/train.all &

# python scripts/reddit/slice_dialogs.py \
#        $original_dir/dev.${max_dialog_turns}turns.src \
#        $original_dir/dev.${max_dialog_turns}turns.tgt \
#        --max-turns $max_dialog_turns \
#        --max-tokens $max_dialog_tokens \
#        < $original_dir/dev.all &

# python scripts/reddit/slice_dialogs.py \
#        $original_dir/test.${max_dialog_turns}turns.src \
#        $original_dir/test.${max_dialog_turns}turns.tgt \
#        --max-turns $max_dialog_turns \
#        --max-tokens $max_dialog_tokens \
#        < $original_dir/test.all &

wait

# paste $original_dir/train.${max_dialog_turns}turns.src $original_dir/train.${max_dialog_turns}turns.tgt | grep -v '<URL>' > $original_dir/train.${max_dialog_turns}turns.nourl.joined &
# paste $original_dir/dev.${max_dialog_turns}turns.src $original_dir/dev.${max_dialog_turns}turns.tgt | grep -v '<URL>' > $original_dir/dev.${max_dialog_turns}turns.nourl.joined & 
# paste $original_dir/test.${max_dialog_turns}turns.src $original_dir/test.${max_dialog_turns}turns.tgt | grep -v '<URL>' > $original_dir/test.${max_dialog_turns}turns.nourl.joined & 
# wait

# cut -f1 $original_dir/train.${max_dialog_turns}turns.nourl.joined > $original_dir/train.${max_dialog_turns}turns.nourl.src &
# cut -f2 $original_dir/train.${max_dialog_turns}turns.nourl.joined > $original_dir/train.${max_dialog_turns}turns.nourl.tgt & 
# cut -f1 $original_dir/dev.${max_dialog_turns}turns.nourl.joined > $original_dir/dev.${max_dialog_turns}turns.nourl.src & 
# cut -f2 $original_dir/dev.${max_dialog_turns}turns.nourl.joined > $original_dir/dev.${max_dialog_turns}turns.nourl.tgt & 
# cut -f1 $original_dir/test.${max_dialog_turns}turns.nourl.joined > $original_dir/test.${max_dialog_turns}turns.nourl.src & 
# cut -f2 $original_dir/test.${max_dialog_turns}turns.nourl.joined > $original_dir/test.${max_dialog_turns}turns.nourl.tgt & 
# wait 

# perl $tokenizer_path < $original_dir/train.${max_dialog_turns}turns.nourl.src | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/_ _ eot _ _/__eot__/g" > $processed_dir/train.${max_dialog_turns}turns.src &
# perl $tokenizer_path < $original_dir/train.${max_dialog_turns}turns.nourl.tgt | sed -e "s/_ _ eou _ _/__eou__/g" > $processed_dir/train.${max_dialog_turns}turns.tgt &

# perl $tokenizer_path < $original_dir/dev.${max_dialog_turns}turns.nourl.src | sed -e "s/_ _ eou _ _/__eou__/g"   | sed -e "s/_ _ eot _ _/__eot__/g" > $processed_dir/dev.${max_dialog_turns}turns.src &
perl $tokenizer_path < $original_dir/dev.${max_dialog_turns}turns.nourl.tgt | sed -e "s/_ _ eou _ _/__eou__/g" > $processed_dir/dev.${max_dialog_turns}turns.tgt &

# perl $tokenizer_path < $original_dir/test.${max_dialog_turns}turns.nourl.src | sed -e "s/_ _ eou _ _/__eou__/g" | sed -e "s/_ _ eot _ _/__eot__/g" > $processed_dir/test.${max_dialog_turns}turns.src &
# perl $tokenizer_path < $original_dir/test.${max_dialog_turns}turns.nourl.tgt | sed -e "s/_ _ eou _ _/__eou__/g"  > $processed_dir/test.${max_dialog_turns}turns.tgt &

# wait
# cat $processed_dir/train.${max_dialog_turns}turns.src > $processed_dir/train.${max_dialog_turns}turns.all
# cat $processed_dir/train.${max_dialog_turns}turns.tgt >> $processed_dir/train.${max_dialog_turns}turns.all


