# Conversational Question Rewriting for Conversational QA

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="img/caire.png" width="20%"> <img align="right" src="img/HKUST.jpeg" width="12%">

The implementation of the paper "Can Question Rewriting Help Conversational Question Answering?":

**Can Question Rewriting Help Conversational Question Answering?**. [Etsuko Ishii](https://etsukokuste.github.io/), [Yan Xu](https://yana-xuyan.github.io), [Samuel Cahyawijaya](https://samuelcahyawijaya.github.io/), Bryan Wilie. **Insights@ACL2022** [[PDF]](https://aclanthology.org/2022.insights-1.13)

If you use any source codes included in this toolkit in your work, please cite the following paper:

<pre>
@inproceedings{ishii-etal-2022-question,
    title = "Can Question Rewriting Help Conversational Question Answering?",
    author = "Ishii, Etsuko  and
      Xu, Yan  and
      Cahyawijaya, Samuel  and
      Wilie, Bryan",
    booktitle = "Proceedings of the Third Workshop on Insights from Negative Results in NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.insights-1.13",
    pages = "94--99",
}
</pre>

## Environment
We implement Python 3.7 and PyTorch 1.10.0, and the other packages follow `requirements.txt`.
Please download dependencies with `pip install -r requirements.txt`.

If you want to log your training with wandb, first install wandb by ```pip install wandb```.
Then, create an account on wandb and log in by ```wandb login```.


## Datasets
You can download all the datasets used in our experiments [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EUeuCXrk_EZEun8jqd9wJb8B0HpxHTTzKSARboyztSbc0w?e=AGvgKt).
Please unzip and place it under `data` directory.

### Question Rewriting Datasets
We use [QReCC dataset (Anantha et al., 2021)](https://github.com/apple/ml-qrecc) and [CANARD dataset (Elgohary et al., 2019)](https://sites.google.com/view/qanta/projects/canard) as the question rewriting datasets.

### Conversational QA Datasets with train-valid-dev split
We use [CoQA (Reddy et al., 2019)](https://stanfordnlp.github.io/coqa/) and [QuAC (Choi et al., 2018)](https://quac.ai) as the conversational question answering datasets.
Since the test set is not publicly available, we randomly sample 5% of dialogues in the training set and adopt them as our validation set and report the test results on the original development set for the CoQA experiments.
Note that the QuAC split is based on [EXCORD (Kim et al., 2021)](https://github.com/dmis-lab/excord), and the CoQA split is done by ourselves.


## Training
### Pretrain a QR model on question rewriting datasets
1. Train a QR model with:
```
CUDA_VISIBLE_DEVICES=0 sh run_qrewrite.sh <dataset: canard|qrecc> <output_dir: save/gpt2-canard|save/gpt2-qrecc> <master_port: 10000>
```
2. Evaluate the QR model with:
```
python infer_qrewrite.py --dataset [canard/qrecc] --exp [gpt2-canard2/gpt2-qrecc] --split validation -cu 0 --overwrite_cache --pretrained_model gpt2 --batchify --eval_bsz 16
```
3. Alternatively, you can download finetuned GPT2 on [QReCC](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/ERz6E09bNepDupKalpMEm8kBlzhHEnNZF2yrpvSAdBFDyA?e=22xsGQ) and [CANARD](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EVHHgq75DcxJoblrKuxlewcBKURAfbkhuNuhFlxwBBU25w?e=D5MnGi) used in our experiments.


### Pretrain a QA model on conversational QA datasets
1. Train a QA model with:
```
CUDA_VISIBLE_DEVICES=0,1 sh run_coqa.sh
```
or, for QuAC, train with:
```
CUDA_VISIBLE_DEVICES=0 sh run_quac.sh  # quac is runnable with only one GPU
```
2. Evaluate the QA model with:
```
sh eval_convqa.sh
```
3. Alternatively, you can download finetuned RoBERTa-base on [CoQA](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EaWslLfuZYVHsPol17FRXDUBusmYhLv77Gc6iHbqJZCyQQ?e=XJzu76) and [QuAC](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EbVZJDy8V4JOiaVTR0i81PQBe6xYm-hq9kVIEviUf0m55Q?e=8lyg6N) used in our experiments.


### Reinforcement learning approach to integrate QR in conversational QA
TBA

### Supervised learning approach to integrate QR in conversational QA
TBA

### Data augmentation approach to integrate QR in conversational QA
TBA