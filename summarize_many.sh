#!/bin/bash
task=translation

src_domain=jesc
tgt_domain=aspec

./summarize.sh $src_domain $tgt_domain $task > exp_logs/$src_domain@$tgt_domain.summary 2> exp_logs/$src_domain@$tgt_domain.score


# src_domain=jesc
# tgt_domain=aspec100k
# ./summarize.sh $src_domain $tgt_domain $task > exp_logs/$src_domain@$tgt_domain.summary 2> exp_logs/$src_domain@$tgt_domain.score


src_domain=opus_it
tgt_domain=opus_acquis
./summarize.sh $src_domain $tgt_domain $task > exp_logs/$src_domain@$tgt_domain.summary 2> exp_logs/$src_domain@$tgt_domain.score


# src_domain=opus_it
# tgt_domain=opus_acquis100k
# ./summarize.sh $src_domain $tgt_domain $task > exp_logs/$src_domain@$tgt_domain.summary 2> exp_logs/$src_domain@$tgt_domain.score
