# Vocabulary Adaptation for Domain Adaptation in Neural Machine Translation

## Download tools
As for requirements for using fairseq, please refer to https://github.com/pytorch/fairseq.
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

* **Setup:** Prepare corpora, subword tokenization, embeddings, and binarized dataset in each domain 
This is an examples of DAfrom JESC to ASPEC for En-Ja translation
```bash
 # Download the dataset of each domain. When trying another dataset, you need to write a script to prepare it by yourself and add the dataset path to 'const.sh'.

 ./scripts/dataset/jesc/setup_dataset.sh
 ./scripts/dataset/aspec/setup_dataset.sh 

 # Train sentencepiece for each domain.
 ./setup_sentencepiece.sh jesc_sp16000.outD.all translation
 ./setup_sentencepiece.sh aspec_sp16000.inD.all translation
 
 # Train CBoW vectors for each domain.
 ./train_cbow.sh jesc_sp16000.outD.all translation
 ./train_cbow.sh aspec_sp16000.inD.all translation  

 # Binarize the datasets for fairseq.
 ./preprocess.sh jesc_sp16000.outD.all translation
 ./preprocess.sh aspec_sp16000.inD.100k translation
```

* **Out-domain:** Train a model by all source domain training set and evaluate it in target domain.
```bash
 ./train.sh jesc_sp16000.baseline translation
 ./preprocess.sh jesc_sp16000@aspec_sp16000.noadapt translation
 ./generate.sh jesc_sp16000@aspec_sp16000.noadapt translation
```

* **In-domain:** Train a model by small target domain fine-tuning set (e.g., 100k) and evaluate it in target domain.
```bash
 ./train.sh aspec_sp16000.inD.100k translation
 ./generate.sh aspec_sp16000.inD.100k translation
```


* **Multi-domain learning (MDL):** Train a model by a concatenation of the source domain training set and the target domain fine-tuning set.
```bash
 ./setup_multidomain_data.sh jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k translation
 ./train_cbow.sh jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k translation
 ./preprocess.sh jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.mdl.domainmixing.100k translation
```

* **Fine-tuning w/ src-domain vocab. (FT-srcV):** Simply apply fine-tuning to the source domain model with no modification (requires the *Out-domain* model trained beforehand).
```bash
 ./train.sh jesc_sp16000@aspec_sp16000.ft.v_jesc_sp16000_all.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.ft.v_jesc_sp16000_all.100k translation
```

* **Fine-tuning w/ tgt-domain vocab. (FT-tgtV):** Train a source-domain model with target-domain vocabulary constructed from target-domain monolingual data.
```bash
 ./train.sh jesc_sp16000.outD.v_aspec_sp16000_100kmono.all translation
 ./train.sh jesc_sp16000@aspec_sp16000.ft.v_aspec_sp16000_100kmono.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.ft.v_aspec_sp16000_100kmono.100k translation
```

* **Finetuning w/ VA-Linear (VA-Linear):** Apply fine-tuning with vocabulary adaptation (by linear transformation) to the source domain model. (requires a *Out-domain* trained beforehand). 
```bash
 ./map_embeddings.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.linear-idt.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.linear-idt.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.linear-idt.100k translation
```

* **Finetuning w/ VA-LLM (VA-LLM):** Apply fine-tuning with vocabulary adaptation (by LLM)to the source domain model. (requires a *Out-domain* trained beforehand). 
```bash
 ./map_embeddings.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
```

## Evaluation and summarization of system outputs
```
mkdir exp_logs
task=translation
src_domain=jesc_sp
tgt_domain=aspec_sp
./generate_many.sh $src_domain $tgt_domain $task
./analyze.sh $src_domain $tgt_domain $task > exp_logs/jesc2aspec.summary 
```

## Optional
* **Data augmentation with back-translation**
```bash
 # Data augmentation part
 ./setup_monolingual_data.sh jesc_sp16000@aspec_sp16000.bt_aug.v_jesc_sp16000_all.100k translation
 ./setup_backtranslation_data.sh jesc_sp16000@aspec_sp16000.bt_aug.v_jesc_sp16000_all.100k translation
 ./preprocess.sh jesc_sp16000@aspec_sp16000.bt_aug.v_jesc_sp16000_all.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.bt_aug.v_jesc_sp16000_all.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.bt_aug.v_jesc_sp16000_all.100k translation

 # Fine-tuning part (w/ src vocab.)
 ./setup_backtranslation_data.sh jesc_sp16000@aspec_sp16000.bt_ft.v_jesc_sp16000_all.100k translation
 ./preprocess.sh jesc_sp16000@aspec_sp16000.bt_ft.v_jesc_sp16000_all.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.bt_ft.v_jesc_sp16000_all.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.bt_ft.v_jesc_sp16000_all.100k translation

 # Fine-tuning part (w/ Vocabulary Adaptation)
 ./setup_backtranslation_data.sh jesc_sp16000@aspec_sp16000.bt_va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
 ./preprocess.sh jesc_sp16000@aspec_sp16000.bt_va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
 ./train.sh jesc_sp16000@aspec_sp16000.bt_va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
 ./generate.sh jesc_sp16000@aspec_sp16000.bt_va.v_aspec_sp16000_100kmono.llm-idt.nn10.100k translation
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
