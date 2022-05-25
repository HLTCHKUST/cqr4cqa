from typing import Dict
from transformers import BatchEncoding

import torch
from torch.utils.data import Dataset


class RewriteDataset(Dataset):
    def __init__(self, data_args, data, tokenizer, model_type="seq2seq"):
        self.args = data_args
        self.sep_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id
        self.data = data
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.model_type == "seq2seq":
            return self.seq2seq_tokenize_function(self.data[index])
        else:
            return self.tokenize_function(self.data[index])

    @staticmethod
    def encoding(tokenizer, sep_token, idx, uttr, input_ids, attention_mask, token_type_ids, label_mask, history_len=3, model_type="seq2seq", tgt=False):
        FLAG = idx % 2 # even 0 odd 1
        if tgt:
            if model_type == "seq2seq":
                uttr = "<bos>"+uttr
            else:
                uttr = "<bos>"+uttr+sep_token
        else:
            if history_len%2 != 0:
                sep_spliter = "<speaker1>" if FLAG==0 else "<speaker2>"
            else:
                sep_spliter = "<speaker2>" if FLAG==0 else "<speaker1>"
            uttr = sep_spliter + uttr

        output = tokenizer.encode(uttr)
        input_ids += output
        if attention_mask is not None:
            attention_mask += [1] * len(output)

        if token_type_ids is not None:
            token_type_ids += [FLAG] * len(output)

        if label_mask is not None:
            if tgt:
                label_mask += [0] + [1] * (len(output)-1)
            else:
                label_mask += [0] * len(output)
        return input_ids, attention_mask, token_type_ids, label_mask
    
    @staticmethod
    def pad(input_ids, attention_mask, token_type_ids, label_mask, pad_token_id, len_to_pad):
        input_padding = [pad_token_id] * len_to_pad
        input_ids = input_ids + input_padding 

        if attention_mask:
            attention_mask_padding = [0] * len_to_pad
            attention_mask = attention_mask + attention_mask_padding 

        if token_type_ids:
            token_type_padding = [0] * len_to_pad
            token_type_ids = token_type_ids + token_type_padding 
        if label_mask:
            label_mask_padding = [0] * len_to_pad
            label_mask = label_mask + label_mask_padding 

        return input_ids, attention_mask, token_type_ids, label_mask

    def tokenize_function(self, example):
        src, tgt = example.split("|||")[0].strip(), example.split("|||")[1].strip()
        src = src.split(self.sep_token)[-self.args.history_len:]
        input_ids = []
        attention_mask = []
        token_type_ids = []
        label_mask = []
        # encode src
        pre_add = self.args.history_len-len(src) 
        assert pre_add >= 0
        for idx, uttr in enumerate(src):
            idx += pre_add
            input_ids, attention_mask, token_type_ids, label_mask = RewriteDataset.encoding(self.tokenizer, self.sep_token, \
            idx, uttr, input_ids, attention_mask, token_type_ids, label_mask, history_len=self.args.history_len, \
            model_type=self.model_type, tgt=False)
            
        # encode tgt
        input_ids, attention_mask, token_type_ids, label_mask = RewriteDataset.encoding(self.tokenizer, self.sep_token, \
        idx+1, tgt, input_ids, attention_mask, token_type_ids, label_mask, history_len=self.args.history_len, \
        model_type=self.model_type, tgt=True)

        # pad or truncate the sequence
        seq_len = len(input_ids) 
        len_to_pad = self.args.max_seq_length - seq_len
        if len_to_pad >= 0 and not self.args.batchify:
            input_ids, attention_mask, token_type_ids, label_mask = RewriteDataset.pad(input_ids, attention_mask, token_type_ids, label_mask, self.pad_token_id, len_to_pad)
        elif len_to_pad < 0:
            # truncate
            input_ids = input_ids[:self.args.max_seq_length]
            attention_mask = attention_mask[:self.args.max_seq_length]
            token_type_ids = token_type_ids[:self.args.max_seq_length]
            label_mask = label_mask[:self.args.max_seq_length]
        
        labels = input_ids.copy()
        label_mask = torch.LongTensor(label_mask)
        labels = torch.LongTensor(labels)
        labels[~label_mask.bool()] = -100

        if not self.args.batchify:
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            token_type_ids = torch.LongTensor(token_type_ids)

            return {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids": token_type_ids, "labels": labels}
        else:
            num_label_mask = label_mask[:].sum()
            input_ids = input_ids[:-num_label_mask]
            attention_mask = attention_mask[:-num_label_mask]
            token_type_ids = token_type_ids[:-num_label_mask]

            return {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids": token_type_ids, "labels": labels.tolist()}
    

    def seq2seq_tokenize_function(self, example):
        src, tgt = example.split("|||")[0].strip(), example.split("|||")[1].strip()
        src = src.split(self.sep_token)[-self.args.history_len:]
        input_ids = []
        attention_mask = []
        # token_type_ids = []
        labels = []
        # label_mask = []
        # encode src
        pre_add = self.args.history_len-len(src) 
        assert pre_add >= 0
        for idx, uttr in enumerate(src):
            idx += pre_add
            input_ids, attention_mask, _, _ = RewriteDataset.encoding(self.tokenizer, self.sep_token, \
            idx, uttr, input_ids, attention_mask, None, None, history_len=self.args.history_len, \
            model_type=self.model_type, tgt=False)
            
        # encode tgt
        labels, _, _, _ = RewriteDataset.encoding(self.tokenizer, self.sep_token, \
        idx+1, tgt, labels, None, None, None, history_len=self.args.history_len, \
        model_type=self.model_type, tgt=True)

        # pad or truncate the sequence
        seq_len_input = len(input_ids) 
        len_to_pad_input = self.args.max_seq_length - seq_len_input
        if len_to_pad_input >= 0:
            input_ids, attention_mask, _, _ = RewriteDataset.pad(input_ids, attention_mask, None, None, self.pad_token_id, len_to_pad_input)
        else:
            # truncate
            input_ids = input_ids[:self.args.max_seq_length]
            attention_mask = attention_mask[:self.args.max_seq_length]
        
        seq_len_output = len(labels) 
        len_to_pad_output = self.args.max_seq_length - seq_len_output
        if len_to_pad_output >= 0:
            labels, _, _, _ = RewriteDataset.pad(labels, None, None, None, self.pad_token_id, len_to_pad_output)
        else:
            labels = labels[:self.args.max_seq_length]
        
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(labels)
        return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": labels}


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
class Seq2SeqDataCollator:
    def __init__(self, pad_token_id, example_id_to_index=None):
        self.pad_token_id = pad_token_id
        self.example_id_to_index = example_id_to_index
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.LongTensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([torch.LongTensor(x["labels"]) for x in batch])

        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)
        decoder_input_ids = labels.clone()[:, :-1]
        labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)[:, 1:]

        mini_batch_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids":decoder_input_ids,
        }

        if "id" in batch[0].keys():  # if "id" in batch: 
            mini_batch_data.update({"id": torch.tensor([self.example_id_to_index[x["id"]] for x in batch])})
        return mini_batch_data

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def ignore_pad_token_for_loss(self, labels, pad_token_id):
        label_mask = labels.eq(pad_token_id)
        labels[label_mask.bool()] = -100
        return labels



def left_pad(seqs, seq_lens, max_len, pad_id):
    padded_seqs = []
    for seq, seq_len in zip(seqs, seq_lens):
        padded_seqs.append(([pad_id] * (max_len - seq_len)) + seq)
    return padded_seqs

def right_pad(seqs, seq_lens, max_len, pad_id):
    padded_seqs = []
    for seq, seq_len in zip(seqs, seq_lens):
        padded_seqs.append(seq + ([pad_id] * (max_len - seq_len)))
    return padded_seqs


class DecoderOnlyCollator:
    def __init__(self, pad_token_id, example_id_to_index=None):
        self.pad_token_id = pad_token_id
        self.example_id_to_index = example_id_to_index

    def __call__(self, features):
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        first = features[0]
        batch = {}

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

        seq_lens, max_len = None, None
        for k, v in first.items():
            if k in ["input_ids", "attention_mask", "token_type_ids"]:
                feat_values = [f[k] for f in features]
                seq_lens = list(map(len, feat_values))
                max_len = max(seq_lens)

                pad_id = 0
                batch[k] = torch.tensor(left_pad(feat_values, seq_lens, max_len, pad_id))
            elif k == "labels":
                # keep using right padding
                feat_values = [f[k] for f in features]
                seq_lens = list(map(len, feat_values))
                max_len = max(seq_lens)

                pad_id = -100
                batch[k] = torch.tensor(right_pad(feat_values, seq_lens, max_len, pad_id))
            elif k in ["id"]:
                batch[k] = torch.tensor([self.example_id_to_index[f[k]] for f in features])
        return batch
    
DataCollators = {
    'seq2seq': Seq2SeqDataCollator,
    'decoder_only': DecoderOnlyCollator
}