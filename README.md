# Label-Reasoning-Network

This repository is the code for "Fine-grained Entity Typing via Label Reasoning" EMNLP2021

We evaluate the performance on two datasets: OntoNotes and Ultra-Fine from [Choi](https://github.com/uwnlp/open_type). Considering the size of the data, we only put the Ultra-Fine in the folder **data/**

| Data                   | Train  | Dev  | Test |
| ---------------------- | ------ | ---- | ---- |
| Ultra-Fine             | 1998   | 1998 | 1998 |
| OntoNotes              | 251039 | 2202 | 8963 |
| OntoNotes augmentation | 793487 | 2202 | 8963 |

## Dependency
> Configure the environment in python=3.7.7：
> ``` bash
> pip install -r requirement.txt
> ```
> then download the [Glove Vector](https://github.com/stanfordnlp/GloVe) and put it in the **data/**, here we use glove.6b.300d.txt

## Test Model
If you want to run the model, download the [mode](https://drive.google.com/drive/folders/1in0OANpIBq6BJC-HNNRVSDlcN1J1oTuq?usp=sharing) and put it into **model/**:

``` bash
bash view_typing.bash
```

## Train Model
1. You can train a LRN w/o IR for example:
``` python
python -m entity_typing.train seq_typing \
  -model-path model/ultra_fine/seq_typing_entity_marker_macro_att1_att2 \
  -label-emb-size 100 \
  -bert-type entity_marker \
  -device 2 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  $*
```

2. You can train a LRN for example:
``` python
python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 3 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -sparce-rate 0.9 \
  $*
```
### others
If you use other Glove Vector, edit the dimension of **-memory-emb-size**

If you want to train another model, you can refer to the **./train_\*.bash** file.

If you want use other attributes, you can add it to data and edit the **process_line** function in **entity_typing/data_utils.py**
# Label-Reasoning-Network
