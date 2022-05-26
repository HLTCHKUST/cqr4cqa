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
Our code for Proximal Policy Optimization (PPO) is modified from [trl](https://github.com/lvwerra/trl).
1. To train the QR model with PPO, run:
```
sh run_ppo.sh
```
You can download trained models: [QReCC+CoQA](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/ESdJ5q8Rrv9NjyZJ54tZyIwBmLplJRxSmDM7XoE-DnC0Yw?e=HrPvW9), [CANARD+CoQA](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EZca-XQCTntPmfTOswm1TTIBhyj59pQQGCWWWFWf2Y41lA?e=LKSG1X), [QReCC+QuAC](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/ESHdoWESsNhFklYkNxKFMigBW7mA0z00LQZfhJaIUfZYFg?e=chJxT8), [CANARD+QuAC](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EV3CcEHWNlBDriRR69V1sOkBNuAtfj0Ex5VjI8eDktdaDQ?e=pZVpBo).

2. To evaluate the trained model, run:
```
sh eval_ppo.sh
```
3. If you want to evaluate models using the metrics reported in the leaderboards, run:
```
python src/modules/convqa_evaluator.py --data coqa --pred-file <path to the model folder>/predictions_test.json --data-file data/coqa/coqa-dev-v1.0.json --out-file <path to the model folder>/all_test_results.txt
```
```
python src/modules/convqa_evaluator.py --data quac --pred-file <path to the model folder>/predictions_test.json --data-file data/quac/val-v0.2.json --out-file <path to the model folder>/all_test_results.txt
```

4. We also support the REINFORCE algorithm to train the QR model. You can run:
```
sh run_reinforce.sh
```
Note that you can evaluate trained models by simply modifying the path to the models of `eval_ppo.sh` even if the QR model is trained with REINFORCE.

### Supervised learning approach to integrate QR in conversational QA
We evaluate with a simple supervised learning approach using rewrites provided by CANARD. 
You can download the QuAC subset that has the CANARD annotations [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EVKnqO3nz7RFqdI2Ltp8IN4BhWmEBUUbFUvSom90pvKXkg?e=x71IoB).
1. To evaluate the CANARD annotations with the [QA model trained on QuAC](https://hkustconnect-my.sharepoint.com/:u:/g/personal/eishii_connect_ust_hk/EbVZJDy8V4JOiaVTR0i81PQBe6xYm-hq9kVIEviUf0m55Q?e=8lyg6N), simply change the path to datasets in `src/data_utils/quac.py` and run:
```
sh eval_convqa.sh
```

2. To train another QA model with the CANARD annotations, change the path to datasets in `src/data_utils/quac.py` in the same way as above, run:
```
sh run_quac.sh
```

### Data augmentation approach to integrate QR in conversational QA
First, we generate 10 possible rewrites using top-k sampling for all the questions of the CQA datasets. 
To guarantee the quality of the rewrites, we select the best F1 scoring ones from every 10 candidates and use them to teach another QR model how to reformulate questions.
1. To generate annotations, run:
```
sh run_augmentation.sh
```
2. Then, train a QR model with `run_qrewrite.py` (refer to `run_qrewrite.sh`) by changing the path to datasets.