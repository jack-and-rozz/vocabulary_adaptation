#!/bin/bash

. ./const.sh

script_dir=$(cd $(dirname $0); pwd)
root_dir=$(pwd)

unprocessed_dir=unprocessed
tokenized_dir=processed.moses
truecased_dir=processed.moses.truecased


###############################
#         En-Ja
###############################
cd $root_dir
if [ ! -e $iwslt_enja_data_root ]; then
    echo "$iwslt_enja_data_root was not found: Download and unpack De-En corpus from the archive (https://wit3.fbk.eu/mt.php?release=2017-01-trnted, https://wit3.fbk.eu/mt.php?release=2017-01-test)"
    exit 1
fi
cd $root_dir/$iwslt_enja_data_root

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


if [ ! -e $unprocessed_dir/train.ja ] || [ ! -e $unprocessed_dir/train.en ]; then
    cat train.tags.en-ja.ja | grep -v "<.\+>" > train.ja.tmp
    cat train.tags.en-ja.en | grep -v "<.\+>" > train.en.tmp

    paste train.ja.tmp train.en.tmp > train.combined.tmp
    rm train.ja.tmp
    rm train.en.tmp
    # Remove empty lines appearing in both languages or either one.
    python -c "[print(l.strip()) for l in open('train.combined.tmp') \
    	          if len(l.split('\t')) == 2 and l.strip()]" > train.filtered
    rm train.combined.tmp
    cut -f1 train.filtered > $unprocessed_dir/train.ja
    cut -f2 train.filtered > $unprocessed_dir/train.en
    rm train.filtered
fi

cd $unprocessed_dir
if [ ! -e dev.en ]; then
    cp IWSLT17.TED.dev2010.en-ja.en dev.en
    cp IWSLT17.TED.dev2010.en-ja.ja dev.ja
fi 
if [ ! -e test.en ]; then
    cp IWSLT17.TED.tst2010.en-ja.en test.en
    cp IWSLT17.TED.tst2010.en-ja.ja test.ja
fi 

cd ..

if [ ! -e $tokenized_dir ]; then
    mkdir $tokenized_dir
fi

files=(train dev test)
for file in ${files[@]}; do
    if [ ! -e $tokenized_dir/$file.en ]; then
	echo "Output tokenized corpora to $tokenized_dir/$file.en"
	perl $root_dir/$tokenizer_path \
	     -l en \
	     < $unprocessed_dir/$file.en \
	     > $tokenized_dir/$file.en &

    fi
    if [ ! -e $tokenized_dir/$file.ja ]; then
	echo "Output tokenized corpora to $tokenized_dir/$file.ja"
	kytea -notags \
	< $unprocessed_dir/$file.ja \
	> $tokenized_dir/$file.ja &
    fi
done
wait 


# Truecasing
if [ ! -e $truecased_dir ]; then
    mkdir -p $truecased_dir
fi

for file in ${files[@]}; do
    if [ ! -e $truecased_dir/$file.en ]; then
	perl $root_dir/$truecaser_script_path \
	     --model $root_dir/$truecaser_model_path.en \
	     < $tokenized_dir/$file.en \
	     > $truecased_dir/$file.en &
    fi
    if [ ! -e $truecased_dir/$file.ja ]; then
	python -c "import unicodedata; [print(unicodedata.normalize('NFKC', s).strip()) for s in open('$tokenized_dir/$file.ja')]" > $truecased_dir/$file.ja &
    fi
done
wait

ln -sf $root_dir/$iwslt_enja_data_root/$truecased_dir/train.en $truecased_dir/monolingual.en
ln -sf $root_dir/$iwslt_enja_data_root/$truecased_dir/train.en $truecased_dir/train.all.en
ln -sf $root_dir/$iwslt_enja_data_root/$tokenized_dir/train.en $tokenized_dir/monolingual.en
ln -sf $root_dir/$iwslt_enja_data_root/$tokenized_dir/train.en $tokenized_dir/train.all.en

ln -sf $root_dir/$iwslt_enja_data_root/$truecased_dir/train.ja $truecased_dir/monolingual.ja
ln -sf $root_dir/$iwslt_enja_data_root/$truecased_dir/train.ja $truecased_dir/train.all.ja
ln -sf $root_dir/$iwslt_enja_data_root/$tokenized_dir/train.ja $tokenized_dir/monolingual.ja
ln -sf $root_dir/$iwslt_enja_data_root/$tokenized_dir/train.ja $tokenized_dir/train.all.ja
