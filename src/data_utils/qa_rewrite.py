import os
import json

from src.data_utils.qr_data_utils import RewriteDataset

file_path_map = {
    "coqa": {
        "train": "train_split.json",
        "validation": "valid_split.json",
        "test": "coqa-dev-v1.0.json",
    },
    "quac":{
        "train": "train.json",
        "validation": "valid.json",
        "test": "val_v0.2.json",
    }
}

def preprocess_coqa(data_file_path, split, output_dir, tokenizer):
    sep_token = tokenizer.eos_token
    with open(data_file_path) as f:
        samples = json.load(f)["data"]

    ids = []
    with open(os.path.join(output_dir,'{}-cache.txt').format(split), 'w') as srch:
        tgt = "unknown"
        for row in samples:
            history = []
            for i, (question, answer) in enumerate(zip(row['questions'], row["answers"])):
                _id = str(row['id']) + '_' + str(question['turn_id'])
                history.append(question['input_text'])
                answer = answer["input_text"]

                src = sep_token.join(history) + "|||" + tgt
                srch.write(src+'\n')

                ids.append(_id)
                history.append(answer)
    
    with open(os.path.join(output_dir,'{}-cache-ids.json').format(split), 'w') as f:
        json.dump(ids, f)


def preprocess_quac(data_file_path, split, output_dir, tokenizer):
    sep_token = tokenizer.eos_token
    with open(data_file_path) as f:
        samples = json.load(f)["data"]

    ids = []
    with open(os.path.join(output_dir,'{}-cache.txt').format(split), 'w') as srch:
        tgt = "unknown"
        for group in samples:
            for item in group['paragraphs']:
                history = []
                for qa in item["qas"]:
                    _id = qa["id"]
                    history.append(qa["question"])
                    answer = qa["orig_answer"]["text"]

                    src = sep_token.join(history) + "|||" + tgt
                    srch.write(src+'\n')

                    # TODO test whether to remove cannotanswer questions
                    ids.append(_id)
                    if answer == "CANNOTANSWER":
                        history.pop()
                    else:
                        history.append(answer)

    with open(os.path.join(output_dir,'{}-cache-ids.json').format(split), 'w') as f:
        json.dump(ids, f)


def load_qa_datasets(data_args, tokenizer, data_dir='data/coqa/', output_dir='data/coqa/', overwrite_cache=False, cache_dir=None, model_type=False):
    data = load_qa_data(data_args.dataset, data_dir=data_dir, output_dir=output_dir, tokenizer=tokenizer, overwrite_cache=overwrite_cache)

    train_dataset = RewriteDataset(data_args, data["train"], tokenizer, model_type=model_type)
    eval_dataset = RewriteDataset(data_args, data["validation"], tokenizer, model_type=model_type)
    if test and data["test"]:
        test_dataset = RewriteDataset(data_args, data["test"], tokenizer, model_type=model_type)
    else:
        test_dataset = None

    datasets = {"train": train_dataset, "validation": eval_dataset, "test":test_dataset}
    return datasets

def load_qa_data(dataset_name, data_dir='data/coqa/', output_dir='data/coqa/', tokenizer=None, overwrite_cache=False, cache_dir=None):
    data = {}

    for split in ['train', 'validation', 'test']:
        if file_path_map[dataset_name][split]:
            cache_file_path = os.path.join(output_dir, f'{split}-cache.txt')

            data_file_path = os.path.join(data_dir, file_path_map[dataset_name][split])
            if not os.path.exists(cache_file_path) or overwrite_cache:
                if "coqa" in dataset_name:
                    preprocess_coqa(data_file_path, split, output_dir, tokenizer)
                elif "quac" in dataset_name:
                    preprocess_quac(data_file_path, split, output_dir, tokenizer)

            with open(cache_file_path, "r") as f:
                lines = f.readlines()
            data[split] = lines
        else:
            data[split] = None
    return data