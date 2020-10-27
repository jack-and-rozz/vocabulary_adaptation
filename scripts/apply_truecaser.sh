# #!/bin/bash
echo "This script is deprecagted."
# usage() {
#     echo "Usage:$0 [aspec-je|jesc-je]"
#     exit 1
# }
# if [ $# -lt 1 ];then
#     usage;
# fi

# data_type=$1
# source_dir=dataset/$data_type/processed.kytea-moses
# target_dir=dataset/$data_type/processed.kytea-moses.truecased

# . ./const.sh

# echo "Source dir: $source_dir, Target dir: $target_dir" 
# if [ -e $source_dir ]; then
#     if [ ! -e $target_dir ]; then
# 	mkdir -p $target_dir
#     fi
#     if [ ! -e $target_dir/train.$src_lang ]; then
# 	perl $truecaser_script_path \
# 	     --model $truecaser_model_path.$src_lang \
# 	     < $source_dir/train.$src_lang \
# 	     > $target_dir/train.$src_lang &
#     fi
#     if [ ! -e $target_dir/dev.$src_lang ]; then
# 	perl $truecaser_script_path \
# 	     --model $truecaser_model_path.$src_lang \
# 	     < $source_dir/dev.$src_lang \
# 	     > $target_dir/dev.$src_lang &
#     fi
#     if [ ! -e $target_dir/test.$src_lang ]; then
# 	perl $truecaser_script_path \
# 	     --model $truecaser_model_path.$src_lang \
# 	     < $source_dir/test.$src_lang \
# 	     > $target_dir/test.$src_lang &
#     fi
#     wait
#     # TODO
#     # if [ ! -e $target_dir/test.$src_lang ] && ; 
#     ln -sf ../$source_dir/train.$tgt_lang $target_dir/train.$tgt_lang
#     ln -sf ../$source_dir/dev.$tgt_lang $target_dir/dev.$tgt_lang
#     ln -sf ../$source_dir/test.$tgt_lang $target_dir/test.$tgt_lang
# else
#     echo '$source_dir does not exist.' 

# fi
