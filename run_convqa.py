"""Trainer & Run Finetuning of LMs on ConvQA datsets"""
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import warnings

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import is_main_process

from src.model.RoBERTamodeling import RobertaForConversationalQuestionAnswering, RobertaForConversationalQuestionAnsweringWithClassHead
from src.modules.convqa_trainer import QuestionAnsweringTrainer
from src.modules.convqa_evaluator import ConvqaEvaluators
from src.data_utils.convqa_data_utils import (
    PROC_DATASET_FUNC,
    load_datasets,
    DataCollatorWithIgnoreComponents
)
from src.utils.utils import add_special_tokens_for_qa
from src.utils.convqa_utils import postprocess_qa_predictions, postprocess_qa_predictions_with_cls

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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
    put_mask_in_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use p mask in the training process."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
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
    eval_coqa: bool = field(
        default=False, metadata={"help": "Evaluation on CoQA dataset, containing yes/no/unknown answers."}
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
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    extra_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the extra datasets to use (via the datasets library). split with '|' "}
    )
    extra_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the extra datasets to use (via the datasets library). split with '|' "}
    )
    add_special_tokens: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add special tokens in the training process"}
    )
    cache_prefix: Optional[str] = field(
        default=None, metadata={"help": "Prefix for the feature cache file to differentiate model"}
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True if data_args.eval_coqa else False,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.eval_coqa:
        model = RobertaForConversationalQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            beta=model_args.beta,
            rational=True,
        )
    else:
        model = RobertaForConversationalQuestionAnsweringWithClassHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if model_args.freeze_emb:
        for p in [
            *model.roberta.embeddings.parameters()
        ]:
            p.requires_grad = False
    
    if data_args.add_special_tokens:
        add_special_tokens_for_qa(model, tokenizer)

    if data_args.eval_coqa:
        training_args.use_fast = True
        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=False,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                )
            training_args.use_fast = False
            warnings.warn(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models \
                at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this requirement",
                UserWarning
            )
    else:
        training_args.use_fast = False
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets, extra_datasets = load_datasets(data_args, model_args.cache_dir)

    train_dataset, validation_dataset, test_dataset, question_column_name, context_column_name, answer_column_name = PROC_DATASET_FUNC["coqa" if data_args.eval_coqa else "quac"](training_args, data_args, datasets, tokenizer, 
                                extra_datasets=extra_datasets, use_token_type_ids="xlnet" in model_args.model_name_or_path, use_classhead=isinstance(model, RobertaForConversationalQuestionAnsweringWithClassHead)) 

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = default_data_collator if data_args.eval_coqa else DataCollatorWithIgnoreComponents()

    # Post-processing:
    def post_processing_function(examples, features, predictions, tokenizer, split="validation", prefix=None):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        postprocess_qa_predictions_func = postprocess_qa_predictions if data_args.eval_coqa else postprocess_qa_predictions_with_cls
        predictions = postprocess_qa_predictions_func(
            examples=examples,
            features=features,
            predictions=predictions,
            tokenizer=tokenizer,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            prefix=prefix,
            is_world_process_zero=trainer.is_world_process_zero(),
            eval_coqa=data_args.eval_coqa,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if data_args.eval_coqa:
            references = [{"id": ex["id"], "answers": ex[answer_column_name], "domain": ex["domain"]} for ex in datasets[split]]
        else:
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets[split]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Initialize early stop callbacks
    callbacks = []
    if model_args.early_stop:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)
        callbacks.append(early_stopping_callback)

    # Cache huggingface datasets data and metric files in model_args.cache_dir
    evaluator = ConvqaEvaluators["coqa"]() if data_args.eval_coqa else ConvqaEvaluators["quac"]()

    def compute_metrics(p):
        return evaluator.compute(p.predictions, p.label_ids) 

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # Aggregate all the arguments and save the arguments for the data
        args = {}
        args.update(vars(model_args))
        args.update(vars(data_args))
        with open(f"{training_args.output_dir}/model_data_args.json", "w") as f:
            json.dump(args, f, indent=4)

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    
    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(
            test_dataset=test_dataset,
            test_examples=datasets["test"],
            metric_key_prefix="test",
        )
        results = test_output.metrics

        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()