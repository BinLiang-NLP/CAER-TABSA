# Introduction
This repository was used in our paper:  
  
**“Context-aware Embedding for Targeted Aspect-based Sentiment Analysis”**  
Bin Liang, Jiachen Du, Ruifeng Xu<sup>*</sup>, Binyang Li, Hejiao Huang. ^*Proceedings of ACL 2019
  
Please cite our paper if you use this code. 

## Requirements

* Python 3.6 / 3.7
* numpy >= 1.13.3
* PyTorch >= 1.0.0

## Usage

### Training
* Train with command, optional arguments could be found in [train.py](/train.py)
* Run refining target: ```./run.sh```


## Model

An overview of our proposed model is given below

<img src="/assets/model.png" width = "50%" />

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper

```bibtex
@inproceedings{liang-etal-2019-context,
    title = "Context-aware Embedding for Targeted Aspect-based Sentiment Analysis",
    author = "Liang, Bin  and
      Du, Jiachen  and
      Xu, Ruifeng  and
      Li, Binyang  and
      Huang, Hejiao",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1462",
    doi = "10.18653/v1/P19-1462",
    pages = "4678--4683",
    abstract = "Attention-based neural models were employed to detect the different aspects and sentiment polarities of the same target in targeted aspect-based sentiment analysis (TABSA). However, existing methods do not specifically pre-train reasonable embeddings for targets and aspects in TABSA. This may result in targets or aspects having the same vector representations in different contexts and losing the context-dependent information. To address this problem, we propose a novel method to refine the embeddings of targets and aspects. Such pivotal embedding refinement utilizes a sparse coefficient vector to adjust the embeddings of target and aspect from the context. Hence the embeddings of targets and aspects can be refined from the highly correlative words instead of using context-independent or randomly initialized vectors. Experiment results on two benchmark datasets show that our approach yields the state-of-the-art performance in TABSA task.",
}
```
