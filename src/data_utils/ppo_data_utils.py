import logging

import torch
from transformers import PreTrainedTokenizerFast

from src.data_utils.convqa_data_utils import prepare_validation_features_fast, convert_examples_to_features
from src.data_utils.qr_data_utils import RewriteDataset

logger = logging.getLogger(__name__)


def proc_dataset_ppo(training_args, data_args, datasets, tokenizer):
    # Preprocessing the datasets
    # currently not supporting batch operation

    sep_token = tokenizer.eos_token
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id
    tokenize_function = TOKENIZATION_FUNC_PPO[training_args.qr_model_type]

    def prepare_features_train(example, idx):
        tokenized_example = tokenize_function(example, idx, tokenizer=tokenizer, sep_token=sep_token, pad_token_id=pad_token_id, \
            history_len=data_args.history_len, max_seq_length=data_args.gen_max_seq_length)
        return tokenized_example

    if training_args.do_train:
        print('run prepare_features_train() for training data...')
        train_dataset = datasets["train"].map(
            prepare_features_train,
            batched=False,
            with_indices=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        train_dataset = None

    def prepare_features_eval(example, idx):
        tokenized_example = tokenize_function(example, idx, tokenizer=tokenizer, sep_token=sep_token, pad_token_id=pad_token_id, \
            history_len=data_args.history_len, max_seq_length=data_args.gen_max_seq_length)
        return tokenized_example

    if training_args.do_train:
        print('run prepare_features_eval() for validation data...')
        validation_dataset = datasets["validation"].map(
                prepare_features_eval,
                batched=False,
                with_indices=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
    else:
        validation_dataset = None

    if training_args.do_eval:
        print('run prepare_features_eval() for test data...')
        test_dataset = datasets["test"].map(
            prepare_features_eval,
            batched=False,
            with_indices=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        test_dataset = None
    return train_dataset, validation_dataset, test_dataset


def tokenize_function_ppo(example, idx, sep_token, pad_token_id, tokenizer, history_len=3, max_seq_length=256):
    src = example["history"][-history_len+1:]
    src.append(example["orig_question"])
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    label_mask = []
    # encode src
    pre_add = history_len-len(src) 
    assert pre_add >= 0
    for idx_, uttr in enumerate(src):
        idx_ += pre_add
        input_ids, attention_mask, token_type_ids, label_mask = RewriteDataset.encoding(tokenizer=tokenizer,
                                                                        sep_token=sep_token,
                                                                        idx=idx_, 
                                                                        uttr=uttr,
                                                                        input_ids=input_ids,
                                                                        attention_mask=attention_mask, 
                                                                        token_type_ids=token_type_ids, 
                                                                        label_mask=label_mask, 
                                                                        model_type="decoder_only",
                                                                        tgt=False)
    # encode tgt (now  tgt = '')
    input_ids, attention_mask, token_type_ids, label_mask = RewriteDataset.encoding(tokenizer=tokenizer,
                                                                        sep_token=sep_token,
                                                                        idx=idx_+1, 
                                                                        uttr='',
                                                                        input_ids=input_ids,
                                                                        attention_mask=attention_mask, 
                                                                        token_type_ids=token_type_ids, 
                                                                        label_mask=label_mask, 
                                                                        model_type="decoder_only",
                                                                        tgt=True)
    # encode gold
    labels, _, _, _ = RewriteDataset.encoding(tokenizer=tokenizer,
                                                                        sep_token=sep_token,
                                                                        idx=idx_+1, 
                                                                        uttr=example["orig_question"],
                                                                        input_ids=[],
                                                                        attention_mask=[], 
                                                                        token_type_ids=[], 
                                                                        label_mask=[], 
                                                                        model_type="decoder_only",
                                                                        tgt=True)

    # pad or truncate the sequence
    seq_len = len(input_ids) 
    len_to_pad = max_seq_length - seq_len
    if len_to_pad < 0:
        # truncate
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        token_type_ids = token_type_ids[:max_seq_length]
        label_mask = label_mask[:max_seq_length]

    # labels = input_ids.copy()
    # labels = labels[1:]

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    label_mask = torch.LongTensor(label_mask)
    labels = torch.LongTensor(labels)
    # labels[~label_mask.bool()] = -100

    num_label_mask = label_mask[:].sum()
    input_ids = input_ids[:-num_label_mask]
    attention_mask = attention_mask[:-num_label_mask]
    token_type_ids = token_type_ids[:-num_label_mask]

    return {"id": example["id"], "input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids": token_type_ids, "labels": labels.tolist(), "domain":example["domain"]}

def seq2seq_tokenize_function_ppo(example, idx, sep_token, pad_token_id, tokenizer, history_len=3, max_seq_length=128):
    src = example["history"][-history_len+1:]
    src.append(example["orig_question"])
    input_ids = []
    attention_mask = []
    labels = []
    # encode src
    pre_add = history_len-len(src) 
    assert pre_add >= 0
    for idx_, uttr in enumerate(src):
        idx_ += pre_add
        input_ids, attention_mask, _, _ = RewriteDataset.encoding(tokenizer=tokenizer,
                                                    sep_token=sep_token,
                                                    idx=idx_, 
                                                    uttr=uttr,
                                                    input_ids=input_ids,
                                                    attention_mask=attention_mask, 
                                                    token_type_ids=None, 
                                                    label_mask=None, 
                                                    model_type="seq2seq",
                                                    tgt=False)
    
    # encode tgt (now  tgt = '')
    labels, _, _, _ = RewriteDataset.encoding(tokenizer=tokenizer,
                                sep_token=sep_token,
                                idx=idx_+1, 
                                uttr=example["orig_question"],
                                input_ids=labels,
                                attention_mask=None, 
                                token_type_ids=None, 
                                label_mask=None, 
                                model_type="seq2seq",
                                tgt=True)

    # pad or truncate the sequence
    seq_len_input = len(input_ids) 
    len_to_pad_input = max_seq_length - seq_len_input
    if len_to_pad_input >= 0:
        pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id
        input_ids, attention_mask, _, _ = RewriteDataset.pad(input_ids, attention_mask, None, None, pad_token_id, len_to_pad_input)
    else:
        # truncate
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
    
    seq_len_output = len(labels) 
    len_to_pad_output = max_seq_length - seq_len_output
    if len_to_pad_output >= 0:
        labels, _, _, _ = RewriteDataset.pad(labels, None, None, None, pad_token_id, len_to_pad_output)
    else:
        labels = labels[:max_seq_length]
    
    # used for cross entropy labels
    shifted_labels = labels[1:]
    
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    shifted_labels = torch.LongTensor(shifted_labels)
    return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": shifted_labels}

TOKENIZATION_FUNC_PPO = {
    "decoder_only": tokenize_function_ppo,
    "seq2seq": seq2seq_tokenize_function_ppo,
}



def proc_dataset_qa(args, dataset, tokenizer, rewrites, use_token_type_ids=False, use_classhead=False, use_history_for_qa=True, is_coqa=False):
    # Preprocessing the datasets for qa evaluation
    column_names = dataset.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    def concat_rewrites(example, rewrites=rewrites, max_query_len=args.max_rewrite_query_length, is_coqa=is_coqa):
        if example["id"] in rewrites:
            # Apply some hacks here for no history
            if not use_history_for_qa:
                example["history"] = []

            example["history"].append(rewrites[example["id"]])
        else:
            raise Exception(f'proc_dataset_qa - key not found | document ID: {example["id"]}')
        # example["history"].append(example["orig_question"])
        question_str = " ".join(list(reversed(example["history"]))).strip() if is_coqa else " ".join(list(reversed(example["history"][-3:]))).strip()
        question = " ".join(question_str.split()[:max_query_len])
        example["question"] = question
        return example

    # Validation preprocessing
    def prepare_validation_features(examples):
        assert is_coqa == True
        tokenized_examples = prepare_validation_features_fast(
            examples,
            tokenizer, 
            context_column_name, 
            question_column_name, 
            answer_column_name, 
            pad_on_right, 
            args.dataset_name, 
            max_seq_length = args.max_seq_length, 
            doc_stride = args.doc_stride, 
            pad_to_max_length = args.pad_to_max_length, 
            use_token_type_ids=use_token_type_ids,
            # use_classhead=use_classhead,
        )
        return tokenized_examples
        
    if not args.orig:
        validation_dataset = dataset.map(
            concat_rewrites,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
        )
    else:
        # Skip concat_rewrites
        validation_dataset = dataset

    if isinstance(tokenizer, PreTrainedTokenizerFast):
        assert is_coqa == True        
        validation_dataset = validation_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
            )
    else:
        assert not is_coqa
        # print('== validation_dataset ==')
        # print(validation_dataset[0])
        # print(validation_dataset[1])
        # print(tokenizer.__class__.__name__)

        if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
            tokenizer.add_prefix_space = True

        validation_dataset_temp = convert_examples_to_features(
            validation_dataset, # Use results from concat_rewrites
            tokenizer=tokenizer,
            data_args=args,
            is_training=False,
            is_coqa="coqa" in args.dataset_name,
            use_classhead=use_classhead,
            threads=args.preprocessing_num_workers,
            tqdm_enabled=False
        )
        validation_dataset = validation_dataset_temp

    return validation_dataset


def prepare_gen_kwargs_for_generation(batch, training_args, ppo_tokenizer, qr_model_type, device):
    input_gen_len = batch["input_ids"].shape[1] if qr_model_type == "decoder_only" else 0
    gen_kwargs = {
        "top_k": training_args.k,
        "top_p": training_args.p,
        "do_sample": training_args.sampling,
        "pad_token_id": ppo_tokenizer.eos_token_id,
        "num_beams": training_args.num_beams,
        "num_beam_groups": 1,
        "temperature": training_args.temperature,
        "max_length": training_args.length + input_gen_len,
        "min_length": 3 + input_gen_len,
        "repetition_penalty": training_args.repetition_penalty,
        "num_return_sequences": training_args.num_return_sequences,
        # "no_repeat_ngram_size": 4
    }
    if "token_type_ids" in batch:
        token_type_ids = batch["token_type_ids"]
        gen_kwargs.update({"token_type_ids": token_type_ids.to(device)})
    
    if qr_model_type == "seq2seq":
        gen_kwargs.update({"decoder_start_token_id": ppo_tokenizer.bos_token_id})
    return gen_kwargs


def prepare_gen_kwargs_for_mixer_generation(batch, training_args, ppo_tokenizer, qr_model_type, device, xent_length):
    input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
    input_gen_len = batch["input_ids"].shape[1] if qr_model_type == "decoder_only" else 0
    gen_kwargs = {
        "top_k": training_args.k,
        "top_p": training_args.p,
        "do_sample": training_args.sampling,
        "pad_token_id": ppo_tokenizer.eos_token_id,
        "num_beams": training_args.num_beams, 
        "num_beam_groups": 1,
        "temperature": training_args.temperature,
        "max_length": training_args.length + input_gen_len,
        "min_length": 3 + input_gen_len,
        "repetition_penalty": training_args.repetition_penalty,
        "num_return_sequences": training_args.num_return_sequences,
        # "no_repeat_ngram_size": 4
    }
    
    # concate gold part into inputs
    labels = batch["labels"][:,:xent_length]
    if "token_type_ids" in batch and qr_model_type == "decoder_only":
        token_type_ids = batch["token_type_ids"][:, :-1]
        gold_token_type_ids = torch.ones(token_type_ids.shape[0], labels.shape[-1], dtype=torch.long)
        token_type_ids = torch.cat([token_type_ids, gold_token_type_ids], dim=-1)
        gen_kwargs.update({"token_type_ids": token_type_ids.to(device)})
    
    if qr_model_type == "seq2seq":
        gen_kwargs.update({
            "decoder_start_token_id": ppo_tokenizer.bos_token_id,
            "decoder_input_ids": labels.to(device),
        })
    else:
        input_ids = torch.cat([input_ids[:, :-1], labels], dim=-1)

        gold_attention_mask = torch.ones(attention_mask.shape[0], labels.shape[-1], dtype=torch.long)
        attention_mask = torch.cat([attention_mask[:, :-1], gold_attention_mask], dim=-1)
    
    return gen_kwargs, input_ids, attention_mask


def prepare_inputs_for_qa(rewrites, batch_ids, features_per_example, index_to_example_id_train, tokenizer, max_question_len=100, args=None, use_token_type_ids=False, use_classhead=False, use_history_for_qa=True, is_coqa=False, is_sc=False):
    examples = {"question": [], "context": [], "answers": [], "id": [], "domain":[]}
    # if len(rewrites) == 2*batch_ids.shape[0], it means the training is in REINFORCE with self-critic mode
    bs = batch_ids.shape[0]
    if len(rewrites) == 2*batch_ids.shape[0] and is_sc:
        batch_ids = batch_ids.repeat(2)
    elif len(rewrites) > batch_ids.shape[0]:
        n = int(len(rewrites) / batch_ids.shape[0])
        batch_ids = torch.repeat_interleave(batch_ids, n)
    else:
        n = 1
    assert len(rewrites) == batch_ids.shape[0]
    count=0
    for i, rew in enumerate(rewrites):
        example_id = index_to_example_id_train[batch_ids[i].item()]
        example = features_per_example[example_id][0]
        # Apply some hacks here for no history
        if use_history_for_qa:
            all_prev_utterances = example["history"]
        else:
            all_prev_utterances = []
        all_prev_utterances.append(rew)
            
        # all_prev_utterances.append(example["orig_question"])
        question_str = " ".join(list(reversed(example["history"]))).strip() if is_coqa else " ".join(list(reversed(example["history"][-3:]))).strip()
        question = " ".join(question_str.split()[:max_question_len])
        example["question"] = question
        if is_sc:
            example["id"] = example_id if i < bs else example_id + "_sc"
        else:
            example["id"] = example_id + '_'*count
            count += 1
            if count == n:
                count = 0
        for k in examples.keys():
            examples[k].append(example[k])

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if isinstance(tokenizer, PreTrainedTokenizerFast):
        assert is_coqa
        batch = prepare_validation_features_fast(examples, tokenizer, context_column_name, question_column_name, answer_column_name, \
                                                pad_on_right, None, max_seq_length=512, doc_stride=128, pad_to_max_length=True, \
                                                use_token_type_ids=use_token_type_ids) #, use_classhead=use_classhead)
    else:
        assert not is_coqa 
        tmp_examples = None
        for k, v_arr in examples.items():
            n = len(v_arr)
            if tmp_examples is None:
                tmp_examples = [{} for _ in range(n)]
            for i, v in enumerate(v_arr):
                tmp_examples[i][k] = v

        # Add prefix space to tokenizer
        if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
            tokenizer.add_prefix_space = True
            
        batch = convert_examples_to_features(tmp_examples, tokenizer=tokenizer, data_args=args, 
                    is_training=False, is_coqa="coqa" in args.dataset_name, 
                    use_classhead=use_classhead, threads=10, tqdm_enabled=False)
        
        # Converts to dict from list of dict
        keys = list(batch[0].keys())
        tmp_batch = {}
        tensor_feats = ['input_ids', 'attention_mask', 'token_type_ids', 'p_mask']

        for key in keys:
            if key in tensor_feats:
                tmp_batch[key] = torch.stack(list(map(lambda x: torch.LongTensor(x[key]), batch)), dim=0)
            else:
                tmp_batch[key] = list(map(lambda x: x[key], batch))
        batch = tmp_batch
        
    example_list = []
    for i in range(len(examples["question"])):
        example_item = {k: v[i] for k, v in examples.items()}
        example_list.append(example_item)
    return batch, example_list