#!/bin/bash
. ./const.sh
original_data_dir=$dataset_root/desc_gen

if [ ! -e $original_data_dir.zip ] && [ ! -e $original_data_dir.zip ]; then
    wget  http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip -O $original_data_dir.zip
fi
if [ ! -e $original_data_dir ]; then
    unzip $dataset_root/desc_gen.zip -d $original_data_dir
    rm $dataset_root/desc_gen.zip
fi


ln -sf valid.eg $original_data_dir/data/slang/dev.eg
ln -sf valid.txt $original_data_dir/data/slang/dev.txt
ln -sf valid.eg $original_data_dir/data/wiki/dev.eg
ln -sf valid.txt $original_data_dir/data/wiki/dev.txt


if [ ! -e $slang_data_root/processed.moses ]; then
    mkdir -p $slang_data_root/processed.moses
fi
if [ ! -e $wikigen_data_root/processed.moses ]; then
    mkdir -p $wikigen_data_root/processed.moses
fi

set_types=(train dev test)
for set_type in ${set_types[@]}; do
    if [ ! -e $slang_data_root/processed.moses/${set_type}.src ]; then
	echo "Applying tokenizer to $dataset_root/desc_gen/data/slang/${set_type}.eg..."

	cut -f2 $dataset_root/desc_gen/data/slang/${set_type}.eg | perl $tokenizer_path | sed -e "s/&lt; TRG &gt;/<TRG>/g"> $slang_data_root/processed.moses/${set_type}.src  &
    fi
    if [ ! -e $slang_data_root/processed.moses/${set_type}.tgt ]; then
	echo "Applying tokenizer to $dataset_root/desc_gen/data/slang/${set_type}.txt..."
	cut -f4 $dataset_root/desc_gen/data/slang/${set_type}.txt | perl $tokenizer_path > $slang_data_root/processed.moses/${set_type}.tgt &
    fi
done
for set_type in ${set_types[@]}; do
    if [ ! -e $wikigen_data_root/processed.moses/${set_type}.src ]; then
	echo "Applying tokenizer to $dataset_root/desc_gen/data/wiki/${set_type}.eg..."

	cut -f2 $dataset_root/desc_gen/data/wiki/${set_type}.eg | perl $tokenizer_path | sed -e "s/&lt; TRG &gt;/<TRG>/g" > $wikigen_data_root/processed.moses/${set_type}.src  &
    fi
    if [ ! -e $wikigen_data_root/processed.moses/${set_type}.tgt ]; then
	echo "Applying tokenizer to $dataset_root/desc_gen/data/wiki/${set_type}.txt..."

	cut -f4 $dataset_root/desc_gen/data/wiki/${set_type}.txt | perl $tokenizer_path > $wikigen_data_root/processed.moses/${set_type}.tgt &
    fi
done
wait


if [ ! -e $wikigen_data_dir ]; then
    mkdir -p $wikigen_data_dir
fi
if [ ! -e $slang_data_dir ]; then
    mkdir -p $slang_data_dir
fi

for set_type in ${set_types[@]}; do
    if [ ! -e $wikigen_data_dir/${set_type}.src ]; then
	echo "Applying truecaser to $wikigen_data_root/processed.moses/${set_type}..."
	perl $truecaser_script_path \
	     --model $truecaser_model_path \
	     < $wikigen_data_root/processed.moses/${set_type}.src \
	    | sed -e 's/<TRG>/ <TRG>/g' \
	    | sed -e 's/^ <TRG>/<TRG>/g' \
		  > $wikigen_data_dir/${set_type}.src &
	perl $truecaser_script_path \
	     --model $truecaser_model_path \
	     < $wikigen_data_root/processed.moses/${set_type}.tgt \
	     > $wikigen_data_dir/${set_type}.tgt &
    fi
    if [ ! -e $slang_data_dir/${set_type}.src ]; then
	echo "Applying truecaser to $slang_data_root/processed.moses/${set_type}..."

	perl $truecaser_script_path \
	     --model $truecaser_model_path \
	     < $slang_data_root/processed.moses/${set_type}.src \
	    | sed -e 's/<TRG>/ <TRG>/g' \
	    | sed -e 's/^ <TRG>/<TRG>/g' \
		  > $slang_data_dir/${set_type}.src &
	perl $truecaser_script_path \
	     --model $truecaser_model_path \
	     < $slang_data_root/processed.moses/${set_type}.tgt \
	     > $slang_data_dir/${set_type}.tgt &
    fi
done;

wait 

if [ ! -e $wikigen_data_dir/train.flat ]; then
    cat $wikigen_data_dir/train.src > $wikigen_data_dir/train.flat
    cat $wikigen_data_dir/train.tgt >> $wikigen_data_dir/train.flat
fi

if [ ! -e $slang_data_dir/train.flat ]; then
    cat $slang_data_dir/train.src > $slang_data_dir/train.flat
    cat $slang_data_dir/train.tgt >> $slang_data_dir/train.flat
fi

for set_type in ${set_types[@]}; do
    if [ ! -e $slang_data_dir/$set_type.word ]; then
	cut -f1 $original_data_dir/data/slang/$set_type.eg > $slang_data_dir/$set_type.word
    fi
    if [ ! -e $wikigen_data_dir/$set_type.word ]; then
	cut -f1 $original_data_dir/data/wiki/$set_type.eg > $wikigen_data_dir/$set_type.word
    fi
done
