import os
import argparse
import logging
import json

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from src.utils.utils import build_compute_metrics_fn_gpt2, remove_v_head, add_special_tokens_
from src.data_utils.canard import load_canard
from src.data_utils.qrecc import load_qrecc
from src.data_utils.qr_data_utils import Seq2SeqDataCollator, DecoderOnlyCollator
from src.data_utils.qa_rewrite import load_qa_datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def Inference(args):
    os.makedirs(f"{args.save_path}/{args.exp}", exist_ok=True)

    # Initialize the model and tokenizer
    model_name_or_path = os.path.join(args.model_folder, args.exp) + "/checkpoint-" + args.checkpoint if args.checkpoint is not None else os.path.join(args.model_folder, args.exp)
    if args.checkpoint is None:
        args.checkpoint = "best"

    config = AutoConfig.from_pretrained(model_name_or_path)    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        reload_special_tokens = False
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        reload_special_tokens = True
    if "gpt2" in args.pretrained_model:
        model, loading_info = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, output_loading_info=True)
        loading_info["unexpected_keys"] = remove_v_head(loading_info["unexpected_keys"])
        assert len(loading_info["missing_keys"]) == len(loading_info["unexpected_keys"])
    else:
        model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config, output_loading_info=True)
        assert len(loading_info["missing_keys"]) == len(loading_info["unexpected_keys"])

    assert len(loading_info["missing_keys"]) == len(loading_info["unexpected_keys"])
    assert len(loading_info["missing_keys"]) == 0
    assert tokenizer.pad_token_id is not None

    if reload_special_tokens:
        add_special_tokens_(model, tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load ckpt
    if args.ckpt != '':
        logger.info("Load the fine-tuned model...")
        model.load_state_dict(torch.load(args.ckpt),strict=False)
    model.to(args.device)

    # get dataloaders
    args.inference = True
    model_type = "decoder_only" if "gpt2" in args.pretrained_model else "seq2seq"
    if args.dataset == 'canard':
        lm_datasets = load_canard(args, tokenizer, overwrite_cache=args.overwrite_cache, model_type=model_type)
    elif args.dataset == 'qrecc':
        lm_datasets = load_qrecc(args, tokenizer, overwrite_cache=args.overwrite_cache, model_type=model_type)
    elif args.dataset == 'coqa' or args.dataset == 'quac':
        lm_datasets = load_qa_datasets(args, tokenizer, data_dir=args.data_dir, output_dir=args.save_path, overwrite_cache=args.overwrite_cache, model_type=model_type)
    else:
        raise ValueError("Invalid dataset!")

    if tokenizer.sep_token is None:
        stop_token = tokenizer.eos_token
    else:
        stop_token = tokenizer.sep_token
    
    print(f"The stop token is {stop_token}")
        
    if os.path.exists(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_generation.txt') and \
        os.path.exists(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_gold.txt') and not args.overwrite and not args.debug:
        print("The result already exists! Skip inference!")
        print("Evaluation starts!")
        Evaluate(args)
        exit()
    
    loader = DataLoader(
        lm_datasets[args.split], 
        batch_size=args.eval_bsz, 
        sampler=SequentialSampler(lm_datasets[args.split]),
        collate_fn=DecoderOnlyCollator(tokenizer.pad_token_id) if (model_type == "decoder_only" and args.batchify) else Seq2SeqDataCollator(tokenizer.pad_token_id),
        shuffle=False
        )

    generated_sequences = []
    golden_sequences = []
    for batch in tqdm(loader, desc=f'Inference', total=len(loader), ncols=100):
        input_gen_len = batch['input_ids'].shape[1] if model_type == "decoder_only" else 0
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        
        gen_kwargs = {
            "top_k": args.k,
            "top_p": args.p,
            "do_sample": args.sampling,
            "pad_token_id": tokenizer.pad_token_id,
            "num_beams": 5, 
            "temperature": args.temperature,
            "max_length": args.length + input_gen_len, 
            "min_length": 5,
            "repetition_penalty": args.repetition_penalty,
        }

        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"]
            gen_kwargs.update({"token_type_ids": token_type_ids.to(args.device)})    
        
        if model_type == "seq2seq":
            gen_kwargs.update({"decoder_start_token_id": tokenizer.bos_token_id})
        
        generated_sequence = model.generate(
            input_ids=input_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
            **gen_kwargs,
        )
        
        for generated_sequence, response in zip(generated_sequence[:, input_gen_len:], batch["labels"]):
            if args.debug:
                logger.info(f"The shape of the output sequences {len(generated_sequence)}.")

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)  # DO NOT skip_special_tokens
            if not args.debug:
                # Remove all text after the stop token
                text = text[: text.find(stop_token) if stop_token and text.find(stop_token)>0 else None]
            generated_sequences.append(text)
            
            response = response[response != -100]
            response_text = tokenizer.decode(response, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            golden_sequences.append(response_text)
            
        if args.debug:
            print(f"The generated sentence is: {text}")
            # print(f"The golden sentence is: {response_text}")
            print("="*80)
            input()

    if not args.debug:
        with open(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_generation.txt', "w") as f:
            for line in generated_sequences:
                f.write(line.replace("\n", " ")+"\n")
        with open(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_gold.txt', "w") as f:
            for line in golden_sequences:
                f.write(line.replace("\n", " ")+"\n")

def Evaluate(args):
    if os.path.exists(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_generation.txt') and os.path.exists(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_gold.txt'):
        print(f"Evaluation generation {args.save_path}/{args.exp}.")

        metric_fn = build_compute_metrics_fn_gpt2("rouge1_recall" if args.dataset == "qrecc" else "bleu")

        preds = []
        with open(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_generation.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                preds.append(line.strip())
        
        golds = []
        with open(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_gold.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                golds.append(line.strip())
        
        results = metric_fn(preds, golds) if args.dataset == "qrecc" else metric_fn(f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_generation.txt', f'{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_gold.txt')

        print(results)
        with open(f"{args.save_path}/{args.exp}/{args.split}_{args.checkpoint}_result.json", "w") as f:
            json.dump(results, f)
    else:
        raise ValueError("Please do inference first!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument(
        "--model_folder",
        default="./save",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
    )
    parser.add_argument('--exp', type=str, default="gpt2-canard")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--eval_bsz", type=int, default=2)
    # generation settings
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    # data settings
    parser.add_argument('--dataset', type=str, default="canard")
    parser.add_argument('--data_dir', type=str, default="./data/canard")
    parser.add_argument('--save_path', type=str, default="./save")
    parser.add_argument('--pretrained_model', type=str, default="gpt2")  
    parser.add_argument('--max_seq_length', type=int, default=256)     # 256 if --history_in_context
    parser.add_argument('--history_len', type=int, default=3)       # history length
    parser.add_argument('--split', type=str, default="test")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='5')
    parser.add_argument('--gold_file', help='gold generation file path', type=str, required=False, default='data/canard/test-gold.txt')
    # parser.add_argument("--add_special_tokens", action="store_true", help="Whether to add special tokens in the input sequence.")

    # for debug
    parser.add_argument("--debug", action="store_true", help="Enter DEBUG mode")
    parser.add_argument("--overwrite", help="Overwrite the inference results even though it exists already", type=bool, default=False)

    parser.add_argument(
        "--preprocessing_num_workers",
        default=None,
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--batchify", action="store_true", help="Prepare the dataset in batch mode."
    )
    args = parser.parse_args()


    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    Inference(args)