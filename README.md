# mderank

This is code for paper: MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction
Data is from [OpenNMT-kpg-release](https://github.com/memray/OpenNMT-kpg-release) and [SIFRank](https://github.com/sunyilgdx/SIFRank).
(Inspec, DUC2001, SemEval2017 are from SIFRank).

## Table of Contents

* [Environment](#installation)
* [Usage](#usage)
* [Cite](#citing-mderank)

## Environment
```
Python 3.7
nltk 3.4.3
StanfordCoreNLP 3.9.1.1
torch 1.1.0
allennlp 0.8.4
pke 1.8.1
transformer 
CUDA version 10.2
```

## Usage
We use run.sh script to run MDERank.
```
sh run.sh
```
--checkpoint is the model used for predictions. Initial MDERank use bert-base-uncased.

## Cite
If you use this code, please cite this paper
```
@article{DBLP:journals/corr/abs-2110-06651,
  author    = {Linhan Zhang and
               Qian Chen and
               Wen Wang and
               Chong Deng and
               Shiliang Zhang and
               Bing Li and
               Wei Wang and
               Xin Cao},
  title     = {MDERank: {A} Masked Document Embedding Rank Approach for Unsupervised
               Keyphrase Extraction},
  journal   = {CoRR},
  volume    = {abs/2110.06651},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.06651},
  eprinttype = {arXiv},
  eprint    = {2110.06651},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-06651.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```