. ./const.sh

root_dir=$(pwd)
script_dir=$(cd $(dirname $0); pwd)
opus_koehn_data_root=$root_dir/$dataset_root/koehn17six
archive_dir=$opus_koehn_data_root/original
bydomain_dir=$opus_koehn_data_root/processed


# They do not distribute the preprocessed setting so the downloading process is omitted. Here we just assume we have preprocessed (splitted, tokenized, and truecased) datasets in 'dataset/koehn17six'.

if [ ! -e $archive_dir ]; then
    echo "$archive_dir was not found. Please manually download the dataset from URL described in the README.md."
    exit 1
fi



# The distributed version was tokenized and truecased in advance.
src_lang=en
tgt_lang=de

domains=(acquis it emea koran subtitles)
langs=($src_lang $tgt_lang)
dtypes=(train dev test)

if [ ! -e $bydomain_dir ]; then
    mkdir $bydomain_dir
fi

# We used preprocessed files (including tokenization and truecasing) made by (Koehn+, '17), so we just create symbolic links in this script to rename data.
for domain in ${domains[@]}; do
    if [ ! -e $bydomain_dir/$domain ] || [ 1=1 ]; then
	mkdir $bydomain_dir/$domain
	for lang in ${langs[@]}; do
	    for dtype in ${dtypes[@]}; do
		ln -sf $archive_dir/$domain-$dtype.tc.$lang \
		   $bydomain_dir/$domain/$dtype.$lang
	    done
	    ln -sf train.$lang $bydomain_dir/$domain/train.all.$lang
	done
    fi
done




