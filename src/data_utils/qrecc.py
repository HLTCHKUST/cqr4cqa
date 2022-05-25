import os
import json

from src.data_utils.qr_data_utils import RewriteDataset

def preprocess(data_file_path, split, output_dir, tokenizer, spacy=True):
    sep_token = tokenizer.eos_token
    with open(data_file_path) as f:
        samples = json.load(f)

    with open(os.path.join(output_dir,'{}-cache.txt').format(split), 'w') as srch:
        for idx, sample in enumerate(samples):
            src = sep_token.join(sample['Context']+[sample['Question']])
            tgt = sample['Rewrite']

            src = src + "|||" + tgt
            srch.write(src+'\n')

def load_qrecc(data_args, tokenizer, data_dir='data/qrecc/', output_dir='data/qrecc/', overwrite_cache=False, model_type="seq2seq"):
    data = load_qrecc_data(data_dir=data_dir, output_dir=output_dir, tokenizer=tokenizer, overwrite_cache=overwrite_cache)

    train_dataset = RewriteDataset(data_args, data["train"], tokenizer, model_type=model_type)
    eval_dataset = RewriteDataset(data_args, data["validation"], tokenizer, model_type=model_type)

    datasets = {"train": train_dataset, "validation": eval_dataset}
    return datasets

def load_qrecc_data(data_dir='data/qrecc/', output_dir='data/qrecc/', tokenizer=None, overwrite_cache=False):
    data = {}
    split_name_map = {'train': 'train', 'test': 'validation'}
    for split in ['train', 'test']:
        cache_file_path = output_dir + split + '-cache.txt'

        data_file_path = data_dir + 'qrecc_' + split + '.json'
        if not os.path.exists(cache_file_path) or overwrite_cache:
            preprocess(data_file_path, split, output_dir, tokenizer)

        with open(cache_file_path) as f:
            lines = f.readlines()
        data[split_name_map[split]] = lines
    return data


if __name__ == '__main__':
    datasets = load_qrecc()