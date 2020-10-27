#!/bin/bash

files=(train generate map_embeddings load_trained_embeddings train_cbow preprocess setup_backtranslation_data setup_sentencepiece setup_multidomain_data setup_monolingual_data tmp101 tmp102 tmp103 tmp199)
for file in ${files[@]}; do
    refresh_sh.sh $file.sh
done
