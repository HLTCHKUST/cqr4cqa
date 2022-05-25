import os
import json

from src.data_utils.qr_data_utils import RewriteDataset

def preprocess(data_file_path, split, output_dir, tokenizer):
    sep_token = tokenizer.eos_token
    with open(data_file_path) as f:
        samples = json.load(f)

    with open(os.path.join(output_dir,'{}-cache.txt').format(split), 'w') as srch:
        for sample in samples:
            src = sep_token.join(sample['History']+[sample['Question']])
            tgt = sample['Rewrite']

            src = src + "|||" + tgt
            srch.write(src+'\n')

def load_canard(data_args, tokenizer, data_dir='data/canard/', output_dir='data/canard/', overwrite_cache=False, model_type="seq2seq"):
    data = load_canard_data(data_dir=data_dir, output_dir=output_dir, tokenizer=tokenizer, overwrite_cache=overwrite_cache)

    train_dataset = RewriteDataset(data_args, data["train"], tokenizer, model_type=model_type)
    eval_dataset = RewriteDataset(data_args, data["validation"], tokenizer, model_type=model_type)
    test_dataset = RewriteDataset(data_args, data["test"], tokenizer, model_type=model_type)

    datasets = {"train": train_dataset, "validation": eval_dataset, "test":test_dataset}
    return datasets

def load_canard_data(data_dir='data/canard/', output_dir='data/canard/', tokenizer=None, overwrite_cache=False):
    data = {}
    split_name_map = {'train': 'train', 'dev': 'validation', 'test': 'test'}
    for split in ['train', 'dev', 'test']:
        cache_file_path = output_dir + split_name_map[split] + '-cache.txt'

        data_file_path = data_dir + split + '.json'
        if not os.path.exists(cache_file_path) or overwrite_cache:
            preprocess(data_file_path, split_name_map[split], output_dir, tokenizer)

        with open(cache_file_path, "r") as f:
            lines = f.readlines()
        data[split_name_map[split]] = lines
    return data


if __name__ == '__main__':
    datasets = load_canard()