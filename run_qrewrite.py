# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
# taken from https://github.com/huggingface/transformers/blob/453a70d4cb99ed4c3febb228c84d2413089e4b57/examples/language-modeling/run_clm.py

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from src.data_utils.canard import load_canard
from src.data_utils.qrecc import load_qrecc
from src.data_utils.qa_rewrite import load_qa_datasets
from src.data_utils.qr_data_utils import Seq2SeqDataCollator
from src.utils.utils import build_compute_metrics_fn_gpt2, add_special_tokens_, postprocess_gpt2_predictions
from src.modules.seq2seq_trainer import GPT2Seq2SeqTrainer

import torch
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "Path to pre-trained model or shortcut name selected"
        },
    )
    model_type: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Model type selected"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="canard", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    history_len: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of history utterances will be concatenated into the inputs."
        },
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Length of the input sequences"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # add_special_tokens: Optional[bool] = field(
    #     default=True, metadata={"help": "Whether to add special tokens in the training process"}
    # )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    batchify: bool = field(
        default=False, metadata={"help": "Prepare the dataset in batch mode."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class ExtraArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls."}
    )        
    from_scratch: Optional[bool] = field(
        default=False,
        metadata={"help": "Train the model from scratch without the pretrained weights."}
    ) 

def main(model_args, data_args, args, extra_args):
    # wandb.init(project="ConvQA", entity='convqa', name=args.output_dir.replace("save/", ""))
    torch.backends.cudnn.deterministic = True

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(args.local_rank) else logging.WARN,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", args)

    # Set seed before initializing model.
    set_seed(args.seed)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    if extra_args.from_scratch:
        logger.info("Train the model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
    else:
        if "gpt2" in model_args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )

    add_special_tokens_(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_type = "decoder_only" if "gpt2" in model_args.model_name_or_path else "seq2seq"
    if data_args.dataset_name == 'canard':
        lm_datasets = load_canard(data_args, tokenizer, overwrite_cache=data_args.overwrite_cache, model_type=model_type)
    elif data_args.dataset_name == 'qrecc':
        lm_datasets = load_qrecc(data_args, tokenizer, overwrite_cache=data_args.overwrite_cache, model_type=model_type)
    elif args.dataset == 'coqa' or args.dataset == 'quac' or args.dataset == 'coqa_split' or args.dataset == 'quac_split':
        lm_datasets = load_qa_datasets(args, tokenizer, data_dir=args.data_dir, output_dir=args.save_path, overwrite_cache=args.overwrite_cache, model_type=model_type)
    else:
        raise ValueError("Invalid dataset!")
    

    # Post-processing:
    def post_processing_function(examples, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, golds = postprocess_gpt2_predictions(
            examples=examples,
            predictions=predictions,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        return predictions, golds

    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=extra_args.early_stopping_patience)
    # nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['Bleu_1', 'Bleu_2', 'Bleu_3', 'ROUGE_L', 'METEOR', 'CIDEr'])  # loads the models
    # metrics_dict = nlgeval.compute_metrics(references, hypothesis)
    compute_metrics_fn = (
        build_compute_metrics_fn_gpt2("rouge1_recall" if data_args.dataset_name == "qrecc" else "bleu") 
    ) if args.predict_with_generate else None
    if "gpt2" in model_args.model_name_or_path:
        trainer = GPT2Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=lm_datasets["train"] if args.do_train else None,
            eval_dataset=lm_datasets["validation"] if args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator, 
            # compute_metrics=compute_metrics_fn,
            # post_process_function=post_processing_function,
            # eval_from_path=data_args.dataset_name == "canard",
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=lm_datasets["train"] if args.do_train else None,
            eval_dataset=lm_datasets["validation"] if args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=Seq2SeqDataCollator(tokenizer.pad_token_id), 
            callbacks=[early_stopping_callback],
        )

    # Training
    if args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(resume_from_checkpoint=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        eval_output["perplexity"] = perplexity

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(eval_output.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, ExtraArguments))
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args, extra_args)