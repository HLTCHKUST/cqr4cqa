from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    qa_model_name_or_path: Optional[str] = field(
        default='/home/etsuko/ConvQA/save/models/roberta-base-coqa-modify', metadata={"help": "pretrained model name or path for QA model"}
    )
    qa_tokenizer_name: Optional[str] = field(
        default='/home/etsuko/ConvQA/save/models/roberta-base-coqa-modify', metadata={"help": "Pretrained tokenizer name or path for QA"}
    )
    ppo_model_name_or_path: Optional[str] = field(
        default='/home/xuyan/ConvQA/save/gpt2-canard', metadata={"help": "pretrained model name or path for Question Rewrites"}
    )
    ppo_tokenizer_name: Optional[str] = field(
        default='/home/xuyan/ConvQA/save/gpt2-canard', metadata={"help": "Pretrained tokenizer name or path for Question Rewrites"}
    )
    adapter: bool = field(
        default=False, metadata={"help": "use adapter structure or not"}
    )
    neck: Optional[int] = field(
        default=256, metadata={"help": "the dimension of the bottleneck of the task adapter"}
    )
    lm: bool = field(
        default=True, metadata={"help": "finetune whole LM or not"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    freeze_emb: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze encoder (if so, we can use large batch size for training)."}
    )
    early_stop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do early stopping in the traning process."}
    )
    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls."}
    )
    beta: int = field(
        default=5,
        metadata={"help": "The hyper-parameter for rationale tagging multi-task training."},
    )
    use_auxiliary_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do use auxiliary loss during training(merge different types of labels together, esecially for yes/no & unk+span prediction)."}
    )
    exclude_yn: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do use auxiliary loss during training(merge different types of labels together, esecially for yes/no & unk+span prediction)."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    orig: Optional[bool] = field(
        default=False, metadata={"help": "Use original question during evaluation"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    extra_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the extra datasets to use (via the datasets library). split with '|' "}
    )
    extra_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the extra datasets to use (via the datasets library). split with '|' "}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    gen_max_seq_length: int = field(
        default=150,
        metadata={
            "help": "The maximum total input sequence length after tokenization for the question rewriting model. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_query_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    gold_data_path: Optional[str] = field(
        default="data/coqa/coqa-dev-v1.0.json", metadata={"help": "The path to the dataset file."}
    )
    rational: bool = field(
        default=False, metadata={"help": "Whether to use rationale tagging task to assit the training process. "}
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=50,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another."
        },
    )
    max_rewrite_query_length: int = field(
        default=100,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will be truncated to this length."
        },
    )
    history_len: int = field(
        default=3, metadata={"help": "keep len of history quesiton-answers"},
    )
    saliency_step: int = field(
        default=15, metadata={"help": "backward steps"},
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class PPOTrainingArguments(TrainingArguments):
    exp: Optional[str] = field(
        default='quac-ppo', metadata={"help": "experiment path to store models and results"},
    )
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "number of ppo epochs"}
    )
    max_save_num: Optional[int] = field(
        default=3, metadata={"help": "the maximum number of models to save"},
    )
    patience: Optional[int] = field(
        default=3, metadata={"help": "patient epochs before early stopping"},
    )
    epochs: Optional[int] = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    batch_size: Optional[int] = field(
        default=2, metadata={"help": "batch size"},
    )
    lr: Optional[float] = field(
        default=1.41e-5, metadata={"help": "ppo learning rate"},
    )
    use_lr_scheduler: Optional[bool] = field(
        default=False, metadata={"help": "use scheduler or not"},
    )
    grad_accum: Optional[int] = field(
        default=1, metadata={"help": "number of gradient accumulation steps"},
    )
    fp16: Optional[bool] = field(default=False)
    # ppo trainer hyper parmeters
    init_kl_coef: Optional[float] = field(default=0.2,)
    adap_kl_ctrl: Optional[bool] = field(default=True,)
    target: Optional[float] = field(default=6.,)
    horizon: Optional[float] = field(default=10000,)
    gamma: Optional[float] = field(default=1.,)
    lam: Optional[float] = field(default=0.95,)
    cliprange: Optional[float] = field(default=0.2,)
    cliprange_value: Optional[float] = field(default=0.2,)
    vf_coef: Optional[float] = field(default=0.1,)
    ce_coef: Optional[float] = field(default=1.0,)
    # generation settings
    length: Optional[int] = field(
        default=200, metadata={"help":"maximum length of generation"},
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help":"temperature of 1.0 has no effect, lower tend toward greedy sampling"},
    )
    repetition_penalty: Optional[float] = field(
        default=1.2, metadata={"help":"primarily useful for CTRL model; in that case, use 1.2"},
    )
    sampling: Optional[bool] = field(default=False,)   
    k: Optional[int] = field(default=0,)
    p: Optional[float] = field(default=0.9,)
    num_beams: Optional[int] = field(default=None)
    num_return_sequences: Optional[int] = field(default=1)
    use_history_for_qa: Optional[bool] = field(default=True)
    # CrossEntropy hyperparameters
    xent_init_length: Optional[int] = field(
        default=10, metadata={"help": "The maxium number of tokens we predict using MIXER in the initalization"}
    )
    xent_min_length: Optional[int] = field(
        default=0, metadata={"help": "The minimum number of tokens we predict using MIXER in the initalization"}
    )
    mixer_update_length_step: Optional[int] = field(
        default=1, metadata={"help": "number of tokens will be shrunken in MIXER training"}
    )
    mixer_steps: Optional[int] = field(
        default=0, metadata={"help": "number of steps between each shrink in MIXER training"}
    )
    use_fast: Optional[bool] = field(
        default=True, metadata={"help": "only fast tokenizers are supported"}
    )


@dataclass
class REINFORCETrainingArguments(TrainingArguments):
    exp: Optional[str] = field(
        default='quac-ppo', metadata={"help": "experiment path to store models and results"},
    )
    max_save_num: Optional[int] = field(
        default=3, metadata={"help": "the maximum number of models to save"},
    )
    patience: Optional[int] = field(
        default=3, metadata={"help": "patient epochs before early stopping"},
    )
    epochs: Optional[int] = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    batch_size: Optional[int] = field(
        default=2, metadata={"help": "batch size"},
    )
    lr: Optional[float] = field(
        default=1.41e-5, metadata={"help": "ppo learning rate"},
    )
    use_lr_scheduler: Optional[bool] = field(
        default=False, metadata={"help": "use scheduler or not"},
    )
    grad_accum: Optional[int] = field(
        default=1, metadata={"help": "number of gradient accumulation steps"},
    )
    fp16: Optional[bool] = field(default=False)
    # generation settings
    length: Optional[int] = field(
        default=200, metadata={"help":"maximum length of generation"},
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help":"temperature of 1.0 has no effect, lower tend toward greedy sampling"},
    )
    repetition_penalty: Optional[float] = field(
        default=1.2, metadata={"help":"primarily useful for CTRL model; in that case, use 1.2"},
    )
    sampling: Optional[bool] = field(default=False,)   
    k: Optional[int] = field(default=0,)
    p: Optional[float] = field(default=0.9,)
    num_beams: Optional[int] = field(default=None)
    num_return_sequences: Optional[int] = field(default=1)
    use_history_for_qa: Optional[bool] = field(default=True)
    # REINFORCE hyperparameters
    reinforce_type: Optional[str] = field(default="with_baseline", metadata={"choices":["with_baseline", "without_baseline", "self_critic"], "help":"which algorithm to use"})
    discount_factor: Optional[float] = field(default=1.0, metadata={"help": "discount factor (default 1.0)"})
    # CrossEntropy hyperparameters
    xent_init_length: Optional[int] = field(
        default=10, metadata={"help": "The maxium number of tokens we predict using MIXER in the initalization"}
    )
    xent_min_length: Optional[int] = field(
        default=0, metadata={"help": "The minimum number of tokens we predict using MIXER in the initalization"}
    )
    mixer_update_length_step: Optional[int] = field(
        default=1, metadata={"help": "number of tokens will be shrunken in MIXER training"}
    )
    mixer_steps: Optional[int] = field(
        default=0, metadata={"help": "number of steps between each shrink in MIXER training"}
    )