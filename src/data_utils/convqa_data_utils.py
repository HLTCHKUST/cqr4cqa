import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import DataProcessor
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from src.utils.data_config import *


logger = logging.getLogger(__name__)

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


class DataCollatorWithIgnoreComponents:
    def __init__(self):
        pass
    
    def __call__(self, features) -> Dict[str, torch.Tensor]:
        """
        modified from HuggingFace default_data_collator
        """

        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids", "tokens", "token_is_max_context", "token_to_orig_map") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def load_datasets(data_args, cache_dir):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    extra_datasets = []
    if data_args.extra_dataset_name is not None:
        dataset_names = data_args.extra_dataset_name.split('|')
        dataset_config_names = data_args.extra_dataset_config_name.split('|')
        for dataset_name, dataset_config_name in zip(dataset_names, dataset_config_names):
            extra_dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
            extra_datasets.append(extra_dataset)

    return datasets, extra_datasets


def prepare_validation_features_fast(examples, tokenizer, context_column_name, question_column_name, answer_column_name, \
                                    pad_on_right, dataset_name, max_seq_length=512, doc_stride=128, pad_to_max_length=True, \
                                    use_token_type_ids=False):

    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=use_token_type_ids,
        padding="max_length" if pad_to_max_length else True,
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
    special_tokens = tokenized_examples.pop("special_tokens_mask")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    tokenized_examples["p_mask"] = [] 
    tokenized_examples["cls_index"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)
        tokenized_examples["cls_index"].append(cls_index)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples["token_type_ids"][i] if use_token_type_ids else tokenized_examples.sequence_ids(i)
        for k, s in enumerate(special_tokens[i]):
            if s:
                sequence_ids[k] = 3
        context_idx = 1 if pad_on_right else 0

        # Build the p_mask: non special tokens and context gets 0.0, the others 1.0.
        tokenized_examples["p_mask"].append(
            [
                0.0 if not special_tokens[i][k] and s == context_idx else 1.0
                for k, s in enumerate(sequence_ids)
            ]
        )

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_idx else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def proc_coqa_dataset(training_args, data_args, datasets, tokenizer, extra_datasets=[], use_token_type_ids=False, **kwargs):
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if not training_args.use_fast:
        processor = DataProcessor(datasets)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if training_args.use_fast:
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_token_type_ids=use_token_type_ids,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            tokenized_examples["cls_idx"] = []
            tokenized_examples["cls_index"] = []
            tokenized_examples["p_mask"] = [] # token type ids is applied to constraint the prediction on context part 
            if "coqa" in data_args.dataset_name:
                tokenized_examples["rational_mask"] = [] 
            
            if "quac" in data_args.dataset_name:
                tokenized_examples["yes_no_labels"] = [] 
                tokenized_examples["followup_labels"] = [] 

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")
            # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
            special_tokens = tokenized_examples.pop("special_tokens_mask")

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)
                tokenized_examples["cls_index"].append(cls_index)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples["token_type_ids"][i] if use_token_type_ids else tokenized_examples.sequence_ids(i)
                for k, s in enumerate(special_tokens[i]):
                    if s:
                        sequence_ids[k] = 3
                context_idx = 1 if pad_on_right else 0

                # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
                # The cls token gets 1.0 too (for predictions of empty answers).
                tokenized_examples["p_mask"].append(
                    [
                        0.0 if not special_tokens[i][k] and s == context_idx else 1.0
                        for k, s in enumerate(sequence_ids)
                    ]
                )

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name if "quac" not in data_args.dataset_name else "orig_answers"][sample_index]
                # Start token index of the current span in the text.
                doc_start_index = 0
                while sequence_ids[doc_start_index] != context_idx:
                    doc_start_index += 1

                # End token index of the current span in the text.
                doc_end_index = len(input_ids) - 1
                while sequence_ids[doc_end_index] != context_idx:
                    doc_end_index -= 1
                
                
                if "coqa" in data_args.dataset_name:
                    # rational mask
                    rational_mask = [0] * len(input_ids)
                    r_start_index = doc_start_index
                    r_end_index = doc_end_index
                    r_start_char = answers["rational_answer_start"][0]
                    r_end_char = answers["rational_answer_end"][0]
                    
                    if r_start_char == -1 or not (offsets[r_start_index][0] <= r_start_char and offsets[r_end_index][1] >= r_end_char):
                        r_start_index = -1
                        r_end_index = -1
                        tokenized_examples["rational_mask"].append(rational_mask)
                    else:
                        while r_start_index < len(offsets) and offsets[r_start_index][0] <= r_start_char:
                            r_start_index += 1
                        r_start_index -= 1
                        while offsets[r_end_index][1] >= r_end_char:
                            r_end_index -= 1
                        r_end_index += 1
                        rational_mask[r_start_index:r_end_index + 1] = [1] * (r_end_index - r_start_index + 1)
                        tokenized_examples["rational_mask"].append(rational_mask)

                if "quac" in data_args.dataset_name:
                    tokenized_examples["yes_no_labels"].append(examples["yesno"][sample_index])
                    tokenized_examples["followup_labels"].append(examples["followup"][sample_index])

                # If no answers are given, set the cls_index as answer.
                if answers["answer_start"][0] == answers["answer_end"][0] == -1 or len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(0)
                    tokenized_examples["end_positions"].append(0)
                    if "coqa" in data_args.dataset_name and r_start_index == -1 and r_end_index == -1: 
                        # leverage rational to see whether yes/no answer is out of span or not
                        tokenized_examples["cls_idx"].append(CLS_UNK)
                    else:
                        # it's really out of span
                        if "yes" in answers["text"][0].lower():
                            tokenized_examples["cls_idx"].append(CLS_YES)
                        elif "no" in answers["text"][0].lower():
                            tokenized_examples["cls_idx"].append(CLS_NO)
                        else:
                            tokenized_examples["cls_idx"].append(CLS_UNK)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    token_start_index = doc_start_index
                    token_end_index = doc_end_index
                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(0)
                        tokenized_examples["end_positions"].append(0)
                        tokenized_examples["cls_idx"].append(CLS_UNK)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)
                        tokenized_examples["cls_idx"].append(CLS_SPAN)
            return tokenized_examples
        else:
            # if tokenizer is not TokenizerFast (e.g., for DeBERTa), we have to go back to the old-school preprocessing
            # print('currently not supporting slow tokenizers')
            raise NotImplementedError

    if training_args.do_train:
        if len(extra_datasets) > 0:
            extra_ds = []
            for d in extra_datasets:
                extra_ds.append(d["train"])
            full_dataset = concatenate_datasets([datasets["train"]]+extra_ds)
        else:
            full_dataset = datasets["train"]

        train_dataset = full_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        train_dataset = None

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if training_args.use_fast:
            tokenized_examples = prepare_validation_features_fast(
                examples,
                tokenizer, 
                context_column_name, 
                question_column_name, 
                answer_column_name, 
                pad_on_right, 
                data_args.dataset_name, 
                max_seq_length = data_args.max_seq_length, 
                doc_stride = data_args.doc_stride, 
                pad_to_max_length = data_args.pad_to_max_length, 
                use_token_type_ids=use_token_type_ids
            )

            return tokenized_examples
        else:
            raise NotImplementedError

    if training_args.do_eval:
        validation_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        validation_dataset = None
    

    # Test preprocessing
    def prepare_test_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if training_args.use_fast:
            tokenized_examples = prepare_validation_features_fast(
                examples,
                tokenizer, 
                context_column_name, 
                question_column_name, 
                answer_column_name, 
                pad_on_right, 
                data_args.dataset_name, 
                max_seq_length = data_args.max_seq_length, 
                doc_stride = data_args.doc_stride, 
                pad_to_max_length = data_args.pad_to_max_length, 
                use_token_type_ids=use_token_type_ids
            )

            return tokenized_examples
        else:
            raise NotImplementedError

    
    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("This dataset doesn't have test set.")

        test_dataset = datasets["test"].map(
            prepare_test_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        test_dataset = None
    return train_dataset, validation_dataset, test_dataset, question_column_name, context_column_name, answer_column_name



def convert_example_to_features(
    example,
    data_args,
    is_training,
    is_coqa=False,
    use_classhead=False,
    ):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # tokenized_examples = defaultdict(list)
    tokenized_examples = {}
    tokenized_examples["input_ids"] = []
    tokenized_examples["attention_mask"] = []
    tokenized_examples["token_type_ids"] = [] 
    tokenized_examples["cls_index"] = [] 
    tokenized_examples["p_mask"] = [] 
    tokenized_examples["tokens"] = []
    tokenized_examples["token_is_max_context"] = [] 
    tokenized_examples["token_to_orig_map"] = [] 
    tokenized_examples["example_id"] = [] 
    if is_training:
        tokenized_examples["start_positions"] = [] 
        tokenized_examples["end_positions"] = [] 
        tokenized_examples["is_impossible"] = [] 
        tokenized_examples["cls_idx"] = []

        if example["answers"]["answer_start"][0] == example["answers"]["answer_end"][0] == -1 or len(example["answers"]["answer_start"]) == 0 or (not is_coqa and example["no_answer"]):
            if "yes" in example["answers"]["text"][0].lower():
                cls_idx = CLS_YES
            elif "no" in example["answers"]["text"][0].lower():
                cls_idx = CLS_NO
            else:
                cls_idx = CLS_UNK
        else:
            cls_idx = CLS_SPAN

        no_answer = example["no_answer"] if not is_coqa else cls_idx == CLS_UNK
    else:
        no_answer = False

    if is_training and not no_answer:
        # Get start and end position
        start_position = example["answers"]["answer_start"][0]
        end_position = example["answers"]["answer_end"][0]

        if not is_coqa:
            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example["context"][start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example["answers"]["text"][0]))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example["context"]):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not no_answer:
        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(example["context"]) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example["answers"]["text"][0]
        )
        if is_coqa:
            r_start_position = example["answers"]["rational_answer_start"][0]
            r_end_position = example["answers"]["rational_answer_end"][0]
            tok_r_start_position = orig_to_tok_index[r_start_position]
            if r_end_position < len(example["context"]) - 1:
                tok_r_end_position = orig_to_tok_index[r_end_position + 1] - 1
            else:
                tok_r_end_position = len(all_doc_tokens) - 1
            (tok_r_start_position, tok_r_end_position) = _improve_answer_span(
                all_doc_tokens, tok_r_start_position, tok_r_end_position, tokenizer, example["answers"]["span_text"][0]
            )

    
    spans = []

    truncated_query = tokenizer.encode(
        example["question"], add_special_tokens=False, truncation=True, max_length=data_args.max_query_length
    )
    if len(truncated_query) == 0:
        print("The question is missing! Add special tokens to avoid bugs!")
        truncated_query = [tokenizer.unk_token_id]
    
    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * data_args.doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus( 
            texts,
            pairs,
            truncation=truncation,
            padding="max_length" if data_args.pad_to_max_length else False,
            max_length=data_args.max_seq_length,
            return_overflowing_tokens=True,
            stride=data_args.max_seq_length - data_args.doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )
        
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * data_args.doc_stride,
            data_args.max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[str(index)] = tok_to_orig_index[len(spans) * data_args.doc_stride + i]
            
        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * data_args.doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * data_args.doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][str(index)] = is_max_context
    
    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # TODO: For one roberta structure, cls_index should be set as 1
        # Set the cls index to 0: the CLS index can be used for impossible answers
        if use_classhead: p_mask[cls_index] = 0

        span_is_impossible = 1.0 if no_answer else 0.0
        start_position = 0
        end_position = 0
        span_cls_idx = CLS_UNK
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if is_coqa and not (tok_r_start_position >= doc_start and tok_r_end_position <= doc_end):
                out_of_span = True
            elif not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = 1.0 # True
                span_cls_idx = CLS_UNK
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                if cls_idx == CLS_SPAN:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                span_cls_idx = cls_idx
        
        tokenized_examples["input_ids"].append(span["input_ids"])
        tokenized_examples["attention_mask"].append(span["attention_mask"])
        tokenized_examples["token_type_ids"].append(span["token_type_ids"])
        tokenized_examples["cls_index"].append(cls_index)
        tokenized_examples["p_mask"].append(p_mask.tolist())
        tokenized_examples["tokens"].append(span["tokens"])
        tokenized_examples["token_is_max_context"].append(span["token_is_max_context"])
        tokenized_examples["token_to_orig_map"].append(span["token_to_orig_map"])
        tokenized_examples["example_id"].append(example["id"])

        if is_training:
            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)
            tokenized_examples["is_impossible"].append(span_is_impossible)
            tokenized_examples["cls_idx"].append(span_cls_idx)

    return tokenized_examples


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(examples, tokenizer, data_args, is_training, is_coqa=False, use_classhead=False, threads=1, tqdm_enabled=True):
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            data_args=data_args,
            is_training=is_training,
            is_coqa=is_coqa,
            use_classhead=use_classhead,
            )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc=f"convert quac examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    
    for example_features in tqdm(
        features, total=len(features), desc="ensemble all the features", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for k in range(len(example_features["input_ids"])):
            new_feature = {}
            for example_feature in example_features:
                new_feature[example_feature] = example_features[example_feature][k]
            new_features.append(new_feature)
    features = new_features
    
    del new_features
    return features


def proc_quac_dataset(training_args, data_args, datasets, tokenizer, extra_datasets=[], use_classhead=True, **kwargs):
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
        tokenizer.add_prefix_space = True
    

    if training_args.do_train:
        if training_args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()
        
        cached_features_file = os.path.join(
            "cache/coqa" if "coqa" in data_args.dataset_name else "cache/quac",
            "cached_transformers_{}_{}_{}_{}".format(
                "train",
                data_args.dataset_config_name,
                data_args.cache_prefix,
                str(data_args.max_seq_length),
                str(data_args.max_query_length)
            ),
        )
        if os.path.exists(cached_features_file) and not data_args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            train_dataset = (
                features_and_dataset["features"]
            )
        else:
            logger.info("Creating features from dataset file at cache/quac")

            if len(extra_datasets) > 0:
                extra_ds = []
                for d in extra_datasets:
                    extra_ds.append(d["train"])
                full_dataset = concatenate_datasets([datasets["train"]]+extra_ds)
            else:
                full_dataset = datasets["train"]

            train_dataset = convert_examples_to_features(
                full_dataset,
                tokenizer=tokenizer,
                data_args=data_args,
                is_training=True,
                is_coqa="coqa" in data_args.dataset_name,
                use_classhead=use_classhead,
                threads=data_args.preprocessing_num_workers,
            )

            if training_args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save({"features": train_dataset}, cached_features_file)
    else:
        train_dataset = None

    
    if training_args.do_eval:
        validation_dataset = convert_examples_to_features(
            datasets["validation"],
            tokenizer=tokenizer,
            data_args=data_args,
            is_training=False,
            is_coqa="coqa" in data_args.dataset_name,
            use_classhead=use_classhead,
            threads=data_args.preprocessing_num_workers,
        )
    else:
        validation_dataset = None


    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("This dataset doesn't have test set.")

        test_dataset = convert_examples_to_features(
            datasets["test"],
            tokenizer=tokenizer,
            data_args=data_args,
            is_training=False,
            is_coqa="coqa" in data_args.dataset_name,
            use_classhead=use_classhead,
            threads=data_args.preprocessing_num_workers,
        )
    else:
        test_dataset = None

    return train_dataset, validation_dataset, test_dataset, question_column_name, context_column_name, answer_column_name

PROC_DATASET_FUNC = {
    "coqa": proc_coqa_dataset,
    "quac": proc_quac_dataset,
}