# 使い方

## Step 1: 各単語の近傍一覧を出す(srcが対象単語の埋め込み，tgtが近傍候補の埋め込み)

```shell
$ python embeddings.py nn [EMB_PATH] > [NN_PATH] 
```

## Step 2: 近傍一覧間の一致度を見る

```shell
$ python embeddings.py merge-nn [NN_PATHS]| python embeddings.py nn-overlap

```

`[NN_PATHS]`はStep 1で獲得した近傍一覧のリスト。近傍一覧の間のすべてのペアに対して、各単語の近傍一覧がどれくらい一致しているかを数える。

出力形式は`jsonl`
