#!/bin/bash

. ./const.sh

script_dir=$(cd $(dirname $0); pwd)
root_dir=$(pwd)

unprocessed_dir=unprocessed
tokenized_dir=processed.moses
truecased_dir=processed.moses.truecased


###############################
#         Fr-En 
##############################

cd $root_dir
if [ ! -e $iwslt_fren_data_root ]; then
    echo "$iwslt_data_root/en-fr was not found: Download and unpack Fr-En corpus from the archive (https://wit3.fbk.eu/mt.php?release=2014-01, https://wit3.fbk.eu/mt.php?release=2014-01-test, https://wit3.fbk.eu/mt.php?release=2014-01-ref)"
    exit 1
fi
cd $root_dir/$iwslt_fren_data_root

if [ ! -e $unprocessed_dir ]; then
    mkdir $unprocessed_dir
fi

for input_file in $(ls *.xml); do
    output_file=$unprocessed_dir/${input_file:0:-4}
    if [ ! -e $output_file ]; then
	echo "Parsing $input_file to $output_file..."

	python $script_dir/parse_xml.py $input_file > $output_file
    fi
done

if [ ! -e $unprocessed_dir/train.fr ] || [ ! -e $unprocessed_dir/train.en ]; then
    cat train.tags.en-fr.fr | grep -v '</' > train.fr.tmp
    cat train.tags.en-fr.en | grep -v '</' > train.en.tmp
    paste train.fr.tmp train.en.tmp > train.combined.tmp
    rm train.fr.tmp
    rm train.en.tmp
    # Remove empty lines appearing in both languages or either one.
    python -c "[print(l.strip()) for l in open('train.combined.tmp') \
    	          if len(l.split('\t')) == 2 and l.strip()]" > train.filtered
    rm train.combined.tmp
    cut -f1 train.filtered > $unprocessed_dir/train.fr
    cut -f2 train.filtered > $unprocessed_dir/train.en
    rm train.filtered
fi

cd $unprocessed_dir
if [ ! -e dev.en ]; then
    cp IWSLT14.TED.dev2010.en-fr.en dev.en
    cp IWSLT14.TED.dev2010.en-fr.fr dev.fr
fi 
if [ ! -e test.en ]; then
    cp IWSLT14.TED.tst2010.en-fr.en test.en
    cp IWSLT14.TED.tst2010.en-fr.fr test.fr
fi 
if [ ! -e test2.en ]; then
    cp IWSLT14.TED.tst2011.en-fr.en test2.en
    cp IWSLT14.TED.tst2011.en-fr.fr test2.fr
fi 
cd ..

if [ ! -e $tokenized_dir ]; then
    mkdir $tokenized_dir
fi

files=(train dev test test2)
suffixes=(en fr)
for file in ${files[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $tokenized_dir/$file.$suffix ]; then
	    perl $root_dir/$tokenizer_path \
		 -l $suffix \
		 < $unprocessed_dir/$file.$suffix \
		 > $tokenized_dir/$file.$suffix &

	fi
    done
done
wait 


# Truecasing
if [ ! -e $truecased_dir ]; then
    mkdir -p $truecased_dir
fi

for file in ${files[@]}; do
    for suffix in ${suffixes[@]}; do
	if [ ! -e $truecased_dir/$file.$suffix ]; then
	    perl $root_dir/$truecaser_script_path \
		 --model $root_dir/$truecaser_model_path.$suffix \
		 < $tokenized_dir/$file.$suffix \
		 > $truecased_dir/$file.$suffix &
	fi
    done
done
wait

