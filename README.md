# Vocabulary Adaptation for Domain Adaptation in Neural Machine Translation

## Requirements
- Python 3.7.3 (other versions can work)
- Sentencepiece 0.1.83 (https://github.com/google/sentencepiece)

## Download tools
```
pip install -r requirements.txt
pip install -r fairseq/requirements.txt

cd tools
git clone https://github.com/moses-smt/mosesdecoder.git 
git clone https://github.com/tmikolov/word2vec.git
git clone https://github.com/jyori112/llm.git
git clone https://github.com/rpryzant/proxy-a-distance.git
cd ..
```

## Reproduction
- This is an example of DA from JESC to ASPEC for En-Ja translation. If you would like to conduct De-En experiments, change "jesc" and "aspec" in the following commands into "opus_it" and "opus_acquis", respectively.
- The scripts used below for our experiments parse a given $model_name (e.g., jesc_sp16000.outD.all) and get parameters related to preprocessing, training, and testing. 


### Setup
```bash
# You first need to manually download datasets from the following URLs and place them to the directories specified in const.sh.
# JESC (En-Ja): https://nlp.stanford.edu/projects/jesc/data/split.tar.gz
# ASPEC (En-Ja): https://jipsti.jst.go.jp/aspec
# OPUS (De-En): https://drive.google.com/file/d/1S48LlMa9RYR9JHQO_KbHdJF8lwVOpLVH/view?usp=sharing

 # Tokenization, truecasing, and placing data to the directory defined by const.sh.
 ./scripts/dataset/jesc/setup_dataset.sh  # En-Ja
 ./scripts/dataset/aspec/setup_dataset.sh # En-Ja
 ./scripts/dataset/koehn17six/setup_dataset.sh # De-En
```

### Training
```
model_name=jesc_sp16000.outD.all
./train.sh $model_name translation

# The model names corresponding to each setting in the original paper (Table 3) are as follows.

# <w/ 100k in-domain parallel data, w/o monolingual data>
# - Out-domain: jesc_sp16000.outD.all (preparing this model is required to train FT-srcV, and VA-*)
# - Out-domain (w/ ASPEC 100k vocab): jesc_sp16000.outD.v_aspec_sp16000_100k.all (preparing this model is required to train FT-tgtV)
# - In-domain : aspec_sp16000.inD.100k

# - MDL: jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k
# - FT-srcV: jesc_sp16000@aspec_sp16000.ft.v_jesc_sp16000_all.100k
# - FT-tgtV: jesc_sp16000@aspec_sp16000.ft.v_aspec_sp16000_100k.100k
# - VA-CBoW: jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100k.nomap.100k
# - VA-Linear: jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100k.linear-idt.100k
# - VA-LLM: jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100k.llm-idt.nn10.100k
```


### Evaluation

```
# When evaluating all models...
mkdir exp_logs
task=translation
src_domain=jesc_sp
tgt_domain=aspec_sp
./generate_many.sh $src_domain $tgt_domain $task
./summarize.sh $src_domain $tgt_domain $task > exp_logs/jesc2aspec.summary

# When evaluating a model
task=translation
model_name=aspec_sp16000.inD.100k
./generate.sh $model_name $task
```



## Citation
If you use this code for research, please cite the following paper.
```
@inproceedings{sato-etal-2020-vocabulary,
    title = "Vocabulary Adaptation for Domain Adaptation in Neural Machine Translation",
    author = "Sato, Shoetsu  and
      Sakuma, Jin  and
      Yoshinaga, Naoki  and
      Toyoda, Masashi  and
      Kitsuregawa, Masaru",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.381",
    pages = "4269--4279",
}
```
