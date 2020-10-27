#!/bin/bash

. ./const.sh

root_dir=$(pwd)
wmt14_data_archive_dir=$(pwd)/$dataset_root/wmt14/archive

if [ ! -e $wmt14_data_archive_dir ]; then
    mkdir -p $wmt14_data_archive_dir
fi
cd $wmt14_data_archive_dir

###################################
#     Download and unpacking
###################################

if [ ! -e training-parallel-europarl-v7.tgz ]; then
    wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz & 
fi

if [ ! -e training-parallel-commoncrawl.tgz ]; then
    wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz & 
fi

if [ ! -e training-parallel-nc-v9.tgz ]; then
    wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz & 
fi

# En-Frは後で
# if [ ! -e training-parallel-un.tgz ]; then
#     wget http://www.statmt.org/wmt13/training-parallel-un.tgz
# fi

# if [ ! -e training-giga-fren.tar ]; then
#     wget http://www.statmt.org/wmt10/training-giga-fren.tar
# fi

if [ ! -e dev.tgz ]; then
   wget http://www.statmt.org/wmt14/dev.tgz & 
fi
# if [ ! -e test-filtered.tgz ];then
#    wget http://www.statmt.org/wmt14/test-filtered.tgz & 
# fi
if [ ! -e test-full.tgz ];then
   wget http://www.statmt.org/wmt14/test-full.tgz & 
fi
wait 


if [ ! -e training/commoncrawl.fr-en.en ]; then
   tar -zxvf training-parallel-commoncrawl.tgz &
   mv commoncrawl.* training
fi
if [ ! -e training/europarl-v7.fr-en.en ]; then
   tar -zxvf training-parallel-europarl-v7.tgz & 
fi

if [ ! -e training/news-commentary-v9.fr-en.en ];then 
   tar -zxvf training-parallel-nc-v9.tgz &
fi
wait

#########################################
#     Construction of each set
#########################################
cd $root_dir
wmt14_deen_data_root=$dataset_root/wmt14/de-en
unprocessed_dir=$(pwd)/$wmt14_deen_data_root/unprocessed
tokenized_dir=$(pwd)/$wmt14_deen_data_root/processed.moses
truecased_dir=$(pwd)/$wmt14_deen_data_root/processed.moses.truecased


# De-En
if [ ! -e $unprocessed_dir ]; then
    mkdir -p $unprocessed_dir
fi

cd $unprocessed_dir
if [ ! -e train.de ];then
    for file in $(ls $wmt14_data_archive_dir/training/*.de-en.de); do
	cat $file >> train.de
    done
fi

if [ ! -e train.en ];then
    for file in $(ls $wmt14_data_archive_dir/training/*.de-en.en); do
	cat $file >> train.en
    done
fi

if [ ! -e dev.de ];then
    for file in $(ls $wmt14_data_archive_dir/dev/*.de); do
	cat $file >> dev.de
    done
fi

if [ ! -e dev.en ];then
    for file in $(ls $wmt14_data_archive_dir/dev/*.en); do
	if [[ ! $file =~ newsdev2014.en ]]; then
	   cat $file >> dev.en
	fi
    done
fi

#####################################################
#     Tokenization and Truecasing to De-En dataset
#####################################################

cd $root_dir
if [ ! -e $tokenized_dir ]; then
    mkdir $tokenized_dir
fi

files=(train dev)
suffixes=(en de)
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

# Use the dev set as testing set for now.
for suffix in ${suffixes[@]}; do
    ln -sf dev.$suffix $truecased_dir/test.$suffix
done

wait


