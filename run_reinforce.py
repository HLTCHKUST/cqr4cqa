import os
import time
import shutil
import collections
from typing import Any, NewType
import logging
import copy
import math
import random
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    EvalPrediction,
    default_data_collator,
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
)
import wandb

from src.data_utils.ppo_data_utils import (
    proc_dataset_ppo, 
    proc_dataset_qa, 
    prepare_inputs_for_qa,
    prepare_gen_kwargs_for_generation,
    prepare_gen_kwargs_for_mixer_generation,
)
from src.data_utils.convqa_data_utils import load_datasets, DataCollatorWithIgnoreComponents
from src.data_utils.qr_data_utils import DataCollators
from src.model.RoBERTamodeling import RobertaForConversationalQuestionAnswering, RobertaForConversationalQuestionAnsweringWithClassHead
from src.model.GPT2modeling import GPT2LMHeadModel
from src.modules.reinforce_trainer import REINFORCETrainer
from src.modules.convqa_evaluator import Evaluator, ConvqaEvaluators
from src.modules.convqa_trainer import QuestionAnsweringTrainer
from src.utils.convqa_utils import postprocess_qa_predictions, postprocess_qa_predictions_with_cls
from src.utils.ppo_args import DataTrainingArguments, ModelArguments, REINFORCETrainingArguments

InputDataClass = NewType("InputDataClass", Any)

logger = logging.getLogger(__name__)

def post_processing_function(examples, 
                            features, 
                            predictions,
                            tokenizer,
                            output_dir=None,
                            stage="eval", 
                            n_best_size=20,
                            max_answer_length=30,
                            eval_coqa=False,
                            ):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    postprocess_qa_predictions_func = postprocess_qa_predictions if eval_coqa else postprocess_qa_predictions_with_cls
    predictions = postprocess_qa_predictions_func(
            examples=examples,
            features=features,
            predictions=predictions,
            tokenizer=tokenizer,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            output_dir=output_dir,
            prefix=stage,
            eval_coqa=eval_coqa,
            batch=True
        )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    if eval_coqa:
        references = [{"id": ex["id"], "answers": ex["answers"], "domain": ex["domain"]} for ex in examples]
    else:
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def compute_f1(predictions):
    pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions.predictions}
    gold_dict = {gold["id"]: gold["answers"]["text"][0] for gold in predictions.label_ids}
    f1_scores = []
    for id_, pred in pred_dict.items():
        f1 = Evaluator.compute_f1(pred, gold_dict[id_])
        f1_scores.append(f1)
    return f1_scores

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(model_args, data_args, training_args):
    # init wandb for logging everything
    wandb.init(project="ConvQA", entity='convqa', name=training_args.exp)
    torch.backends.cudnn.deterministic = True

    # init seed
    set_seed(0)

    """
    Load QA Model
    """
    is_coqa = True if 'coqa' in data_args.dataset_name else False
    qa_config = AutoConfig.from_pretrained(
        model_args.qa_model_name_or_path,
        cache_dir=data_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if is_coqa:
        qa_model = RobertaForConversationalQuestionAnswering.from_pretrained(
                model_args.qa_model_name_or_path,
                from_tf=bool(".ckpt" in model_args.qa_model_name_or_path),
                config=qa_config,
                cache_dir=data_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                beta=model_args.beta,
                rational=True,
            )
    else:
        qa_model = RobertaForConversationalQuestionAnsweringWithClassHead.from_pretrained(
                model_args.qa_model_name_or_path,
                from_tf=bool(".ckpt" in model_args.qa_model_name_or_path),
                config=qa_config,
                cache_dir=data_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        model_args.qa_tokenizer_name, 
        use_fast=True if is_coqa else False,
        model_max_len=data_args.max_seq_length
    )

    """
    Load RL Model
    """
    ppo_tokenizer = AutoTokenizer.from_pretrained(model_args.ppo_tokenizer_name)
    config = AutoConfig.from_pretrained(model_args.ppo_model_name_or_path)
    qr_model_type = "decoder_only" if "gpt2" in model_args.ppo_model_name_or_path else "seq2seq"
    training_args.qr_model_type = qr_model_type
    if qr_model_type == "decoder_only":
        config.adapter = model_args.adapter
        config.lm = model_args.lm
        config.neck = model_args.neck if model_args.adapter else None
        qr_model = GPT2LMHeadModel.from_pretrained(model_args.ppo_model_name_or_path, config=config)
    else:
        qr_model = AutoModelForSeq2SeqLM.from_pretrained(model_args.ppo_model_name_or_path, config=config)
    # wandb.watch(qr_model, log='all')

    # Set device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    qa_model.to(device)
    qr_model.to(device)

    for param in qa_model.parameters():
        param.requires_grad = False

    # Set up QA evaluator
    evaluator = ConvqaEvaluators["coqa"if is_coqa else "quac"]()
    def compute_metrics(p):
        return evaluator.compute(p.predictions, p.label_ids) 
    # Post-processing function for QA Evaluation
    def post_processing_function_eval(examples, features, predictions, tokenizer, split="validation", prefix=None, *args, **kwargs):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        postprocess_qa_predictions_func = postprocess_qa_predictions if is_coqa else postprocess_qa_predictions_with_cls
        predictions = postprocess_qa_predictions_func(
            examples=examples,
            features=features,
            predictions=predictions,
            tokenizer=tokenizer,
            n_best_size=20,
            max_answer_length=data_args.max_answer_length,
            output_dir=os.path.join(training_args.output_dir, training_args.exp),
            prefix=prefix,
            is_world_process_zero=True,
            eval_coqa=is_coqa,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if is_coqa:
            references = [{"id": ex["id"], "answers": ex["answers"], "domain": ex["domain"]} for ex in datasets[split]]
        else:
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets[split]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    """
    Load Dataset
    """
    datasets, _ = load_datasets(data_args, data_args.cache_dir)
    train_dataset, validation_dataset, test_dataset = proc_dataset_ppo(training_args, data_args, datasets, ppo_tokenizer)
    
    """
    Training & Validation
    """
    if training_args.do_train:
        print('== DO TRAINING ==')
        
        # Process Training Data
        example_id_to_index_train = {k: i for i, k in enumerate(train_dataset["id"])}
        index_to_example_id_train = {i: k for i, k in enumerate(train_dataset["id"])}
        
        train_dataset.set_format(output_all_columns=True)
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(train_dataset):
            example = {"history": feature["history"],
                        "context": feature["context"],
                        "answers": feature["answers"],
                        "domain": feature["domain"] if is_coqa else None,
            }
            features_per_example[feature["id"]].append(example)

        # Process Validation Data
        example_id_to_index_validation = {k: i for i, k in enumerate(validation_dataset["id"])}
        index_to_example_id_validation = {i: k for i, k in enumerate(validation_dataset["id"])}

        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_args.batch_size, 
            sampler=SequentialSampler(train_dataset),
            collate_fn=DataCollators[qr_model_type](pad_token_id=ppo_tokenizer.pad_token_id, example_id_to_index=example_id_to_index_train),
        )
        num_update_steps_per_epoch = len(train_loader) // training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(training_args.epochs * num_update_steps_per_epoch)
        training_args.num_training_steps = max_steps

        # initialize mixer steps
        if training_args.mixer_steps > 0:
            xent_length = max(training_args.xent_init_length, 0)
            print(f"The mixer training will be in the first {training_args.mixer_steps / len(train_loader) * training_args.grad_accum} epochs!")
        else:
            xent_length = 0

        qr_loader = DataLoader(
                validation_dataset, 
                batch_size=training_args.per_device_eval_batch_size, 
                sampler=SequentialSampler(validation_dataset),
                collate_fn=DataCollators[qr_model_type](pad_token_id=ppo_tokenizer.pad_token_id, example_id_to_index=example_id_to_index_validation),
        )
        reinforce_trainer = REINFORCETrainer(qr_model, training_args)
        best_scores = {'eval_exact_match': 0, 'eval_f1': 0, 'epoch': -1}
        save_paths = []
        patience = 0
        global_step = 0
        for epoch in range(training_args.epochs):
            print(f'=== TRAINING | epoch {epoch} ===')
            for batch_id, batch in enumerate(tqdm(train_loader)):

                print(f'=== batch_id | {batch_id} ===')
                # torch.cuda.empty_cache()
                torch.set_grad_enabled(False)

                qr_model.eval()
                qa_model.eval()
                
                logs = dict()
                game_data = dict()
                timing = dict()
                t0 = time.time()
                
                #### get response from gpt2  
                rewrites = []
                bid = batch['id']
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                token_type_ids = batch["token_type_ids"] if qr_model_type == "decoder_only" else None
                ce_labels = batch["labels"][:,1:xent_length+1]
                input_gen_len = input_ids.shape[1] if qr_model_type == "decoder_only" else 0

                # Update length for cross_entropy in MIXER training
                if training_args.mixer_steps > 0:
                    if reinforce_trainer.is_update_step():
                        global_step += 1
                    if global_step >= training_args.mixer_steps:
                        xent_length = max(xent_length-training_args.mixer_update_length_step, training_args.xent_min_length)
                        global_step = 0

                if xent_length > 0:
                    # use mixer training
                    gen_kwargs, mixer_input_ids, mixer_attention_mask = prepare_gen_kwargs_for_mixer_generation(batch, training_args, ppo_tokenizer, qr_model_type, device, xent_length)
                    
                    # Replace -100 to valid index
                    mixer_input_ids[mixer_input_ids == -100] = ppo_tokenizer.pad_token_id
                    if 'decoder_input_ids' in gen_kwargs:
                        gen_kwargs['decoder_input_ids'][gen_kwargs['decoder_input_ids'] == -100] = ppo_tokenizer.pad_token_id
                    
                    generated_sequences = qr_model.generate(
                        input_ids=mixer_input_ids.to(device),
                        attention_mask=mixer_attention_mask.to(device),
                        **gen_kwargs,
                    )
                else:
                    gen_kwargs = prepare_gen_kwargs_for_generation(batch, training_args, ppo_tokenizer, qr_model_type, device)
                    generated_sequences = qr_model.generate(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        **gen_kwargs,
                    )

                for seq in generated_sequences:
                    rewrite = ppo_tokenizer.decode(seq[input_gen_len:], skip_special_tokens=True)
                    # rewrite = rewrite[:rewrite.find(ppo_tokenizer.eos_token)]
                    rewrites.append(rewrite)

                print(rewrites)
                game_data['response'] = rewrites
                timing['time/get_response'] = time.time()-t0

                if training_args.reinforce_type == "self_critic":
                    # get qr generation with greedy decoding
                    gen_kwargs["do_sample"] = False
                    gen_kwargs["num_beams"] = 1

                    if xent_length > 0:
                        # use mixer training
                        gen_kwargs, mixer_input_ids, mixer_attention_mask = prepare_gen_kwargs_for_mixer_generation(batch, training_args, ppo_tokenizer, qr_model_type, device, xent_length)
                        
                        # Replace -100 to valid index
                        mixer_input_ids[mixer_input_ids == -100] = ppo_tokenizer.pad_token_id
                        if 'decoder_input_ids' in gen_kwargs:
                            gen_kwargs['decoder_input_ids'][gen_kwargs['decoder_input_ids'] == -100] = ppo_tokenizer.pad_token_id
                        
                        generated_sequences_sc = qr_model.generate(
                            input_ids=mixer_input_ids.to(device),
                            attention_mask=mixer_attention_mask.to(device),
                            **gen_kwargs,
                        )
                    else:
                        gen_kwargs = prepare_gen_kwargs_for_generation(batch, training_args, ppo_tokenizer, qr_model_type, device)
                        if 'decoder_input_ids' in gen_kwargs:
                            gen_kwargs['decoder_input_ids'][gen_kwargs['decoder_input_ids'] == -100] = ppo_tokenizer.pad_token_id                        
                        generated_sequences_sc = qr_model.generate(
                            input_ids=input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            **gen_kwargs,
                        )

                    for seq in generated_sequences_sc:
                        rewrite = ppo_tokenizer.decode(seq[input_gen_len:], skip_special_tokens=True)
                        # rewrite = rewrite[:rewrite.find(ppo_tokenizer.eos_token)]
                        rewrites.append(rewrite)

                #### tokenize text for qa
                t = time.time()
                qa_batch, qa_example = prepare_inputs_for_qa(rewrites, bid, features_per_example, index_to_example_id_train, qa_tokenizer, args=data_args, use_token_type_ids = "xlnet" in model_args.qa_model_name_or_path, use_classhead=isinstance(qa_model, RobertaForConversationalQuestionAnsweringWithClassHead), use_history_for_qa=training_args.use_history_for_qa, is_coqa=is_coqa)
                timing['time/build_input_qa'] = time.time()-t
                #### get F1 score as reward
                t = time.time()
                
                outputs = qa_model(input_ids=torch.LongTensor(qa_batch["input_ids"]).to(device),
                                    attention_mask=torch.LongTensor(qa_batch["attention_mask"]).to(device),
                                    p_mask=torch.LongTensor(qa_batch["p_mask"]).to(device),
                                    return_dict=True)
                if is_coqa:
                    predictions = (outputs.start_logits.detach().cpu().numpy(), 
                                        outputs.end_logits.detach().cpu().numpy(),
                                        outputs.yes_logits.detach().cpu().numpy(),
                                        outputs.no_logits.detach().cpu().numpy(),
                                        outputs.unk_logits.detach().cpu().numpy())
                else:
                    predictions = (outputs.start_logits.detach().cpu().numpy(), 
                                        outputs.end_logits.detach().cpu().numpy(),
                                        None)
                qa_batch_list = []
                for i in range(len(qa_batch["input_ids"])):
                    qa_batch_item = {k: v[i] for k, v in qa_batch.items()}
                    qa_batch_list.append(qa_batch_item)
                prediction = post_processing_function(qa_example, qa_batch_list, predictions, qa_tokenizer, eval_coqa=is_coqa)
                f1_scores = compute_f1(prediction)
                    
                rewards = torch.FloatTensor(f1_scores)
                # rewards[rewards == 0] = -0.1
                # print('rewards', rewards)
                
                timing['time/get_qa_preds'] = time.time()-t

                #### Run REINFORCE training 
                t = time.time()
                
                torch.set_grad_enabled(True)                
                qr_model.train()
                
                # calculate right padding
                num_right_pad = (generated_sequences == ppo_tokenizer.eos_token_id).sum(dim=-1)
                ns = training_args.num_return_sequences
                if qr_model_type == 'decoder_only':
                    gen_length = generated_sequences.shape[-1] - input_ids.shape[-1]
                    gen_attention_mask = torch.ones(generated_sequences.shape[0], gen_length, dtype=torch.long)
                    for i, rp in enumerate(num_right_pad):
                        if rp != 0:
                            gen_attention_mask[i,-rp:] = 0
                    attention_mask = torch.repeat_interleave(attention_mask, ns, dim=0)
                    attention_mask = torch.cat([attention_mask, gen_attention_mask], dim=-1)
                
                    # Generate PPO token type id
                    gen_token_type_ids = torch.ones(generated_sequences.shape[0], gen_length, dtype=torch.long)
                    token_type_ids = torch.repeat_interleave(token_type_ids, ns, dim=0)
                    token_type_ids = torch.cat([token_type_ids, gen_token_type_ids], dim=-1)
                
                    # Generate QR Input
                    qr_inputs = qr_model.prepare_inputs_for_generation(generated_sequences, 
                                attention_mask=attention_mask, token_type_ids=token_type_ids)                

                    ce_labels = torch.repeat_interleave(ce_labels, ns, dim=0)
                    stats = reinforce_trainer.step(
                            qr_inputs['input_ids'].to(device), qr_inputs['position_ids'].to(device), qr_inputs['attention_mask'].to(device),
                            qr_inputs['token_type_ids'].to(device), gen_length, rewards.to(device), num_ce_tokens=ce_labels.shape[-1], ce_labels=ce_labels
                    )
                else:
                    gen_length = generated_sequences.shape[-1]
                    input_ids = torch.repeat_interleave(input_ids, ns, dim=0).to(device)
                    gen_attention_mask = (generated_sequences != ppo_tokenizer.eos_token_id).long().roll(1, dims=-1)
                    gen_attention_mask[:,0] = 1 # Fill first attention mask to 1
                    attention_mask = torch.repeat_interleave(attention_mask, ns, dim=0).to(device)
                    ce_labels = torch.repeat_interleave(ce_labels, ns, dim=0)
                    
                    stats = reinforce_trainer.step(
                            input_ids.to(device), None, attention_mask.to(device), None, gen_length, rewards.to(device), 
                            decoder_input_ids=generated_sequences.to(device), decoder_attention_mask=gen_attention_mask.to(device),
                            num_ce_tokens=ce_labels.shape[-1], ce_labels=ce_labels
                    )
                del generated_sequences, input_ids, input_gen_len, attention_mask, token_type_ids 
                del outputs, qa_batch, qa_example, predictions, prediction
                del gen_length
                
                timing['time/optimization'] = time.time()-t
                    
                #### Log everything
                timing['time/epoch'] = time.time()-t0
                
                table_rows = [list(r) for r in zip(game_data['response'], rewards.cpu().tolist())]
                logs.update({'game_log':wandb.Table(
                    columns=['response', 'reward'],
                        rows=table_rows)})
                logs.update(timing)
                # logs.update(stats)
                logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
                logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
                logs['env/reward_dist'] = rewards.cpu().numpy()
                wandb.log(logs)
                    
                # Cleans up everything meticulously just like how Ishii cleans up the lab
                del rewards, stats, logs

            # epoch-wise validation on validation set
            print(f'=== VALIDATION | epoch {epoch} ===')
            qr_model.eval()
            rewrites = {}
            for i, batch in enumerate(tqdm(qr_loader)):
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                input_gen_len = input_ids.shape[1] if qr_model_type == "decoder_only" else 0

                gen_kwargs = prepare_gen_kwargs_for_generation(batch, training_args, ppo_tokenizer, qr_model_type, device)
                gen_kwargs["num_return_sequences"] = 1
                generated_sequences = qr_model.generate(
                        input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        **gen_kwargs,
                    )
                for seq, ex_id in zip(generated_sequences, batch["id"]):
                    rewrite = ppo_tokenizer.decode(seq[input_gen_len:], skip_special_tokens=True)
                    # rewrite = rewrite[:rewrite.find(ppo_tokenizer.eos_token)]
                    rewrites[index_to_example_id_validation[ex_id.item()]] = rewrite

            # now all the rewrites are stored in dict
            # let's do QA loop
            validation_dataset = proc_dataset_qa(data_args, datasets['validation'], qa_tokenizer, rewrites, use_token_type_ids = "xlnet" in model_args.qa_model_name_or_path, use_history_for_qa=training_args.use_history_for_qa, is_coqa=is_coqa)
            os.makedirs(f"{training_args.output_dir}/{training_args.exp}/", exist_ok=True)

            qa_training_args = copy.deepcopy(training_args)
            qa_training_args.report_to = ['tensorboard']
            trainer = QuestionAnsweringTrainer(
                        model=qa_model,
                        args=qa_training_args,
                        train_dataset=None,
                        eval_dataset=validation_dataset,
                        eval_examples=datasets["validation"],
                        tokenizer=qa_tokenizer,
                        data_collator=default_data_collator if is_coqa else DataCollatorWithIgnoreComponents(),
                        post_process_function=post_processing_function_eval,
                        compute_metrics=compute_metrics
                    )
            print("*** VALIDATION | Run Trainer.Evaluate ***")
            results = trainer.evaluate()
            print()
            print(f"EPOCH {epoch} F1:{results['eval_f1']} EM: {results['eval_exact_match']}")
            print()

            output_eval_file = os.path.join(training_args.output_dir, training_args.exp, "eval_results_"+str(epoch)+".txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    print("***** Validation results *****")
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
                   
            # save models if it outperforms 
            if results["eval_exact_match"] + results["eval_f1"] > best_scores["eval_exact_match"] + best_scores["eval_f1"]:
                patience = 0
                best_scores["eval_exact_match"] = results["eval_exact_match"]
                best_scores["eval_f1"] = results["eval_f1"]
                best_scores["epoch"] = epoch
                os.makedirs(f"{training_args.output_dir}/{training_args.exp}/epoch-{str(epoch)}", exist_ok=True)
                save_path = os.path.join(training_args.output_dir, training_args.exp, 'epoch-'+str(epoch))
                save_paths.append(save_path)
                qr_model.save_pretrained(save_path)
                ppo_tokenizer.save_pretrained(save_path)
                if len(save_paths) > training_args.max_save_num:
                    shutil.rmtree(save_paths.pop(0))
            else:
                patience += 1
            if patience >= training_args.patience:
                print("out of patience!")
                exit(1)

    """
    Evaluation
    """
    if training_args.do_eval:
        print('== DO EVALUATION ==')
        if os.path.exists(os.path.join(training_args.output_dir, training_args.exp, "eval_results.txt")):
            exit(1)
        
        # evaluation!
        # In evaluation, first we get all the rewrites, and then we do all the QA (not batch-by-batch)
        example_id_to_index_test = {k: i for i, k in enumerate(test_dataset["id"])}
        index_to_example_id_test = {i: k for i, k in enumerate(test_dataset["id"])}

        qr_loader = DataLoader(
            test_dataset, 
            batch_size=training_args.per_device_eval_batch_size, 
            sampler=SequentialSampler(test_dataset),
            collate_fn=DataCollators[qr_model_type](pad_token_id=ppo_tokenizer.pad_token_id, example_id_to_index=example_id_to_index_test),
            )
        rewrites = {}
        for batch in tqdm(qr_loader, desc=f'QR Inference', total=len(qr_loader), ncols=100):
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_gen_len = input_ids.shape[1] if qr_model_type == "decoder_only" else 0

            gen_kwargs = prepare_gen_kwargs_for_generation(batch, training_args, ppo_tokenizer, qr_model_type, device)
            gen_kwargs["num_return_sequences"] = 1
            generated_sequences = qr_model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    **gen_kwargs,
            )
            for seq, ex_id in zip(generated_sequences, batch["id"]):
                rewrite = ppo_tokenizer.decode(seq[input_gen_len:], skip_special_tokens=True)
                # rewrite = rewrite[:rewrite.find(ppo_tokenizer.eos_token)]
                rewrites[index_to_example_id_test[ex_id.item()]] = rewrite

        # now all the rewrites are stored in dict
        # let's do QA loop
        test_dataset = proc_dataset_qa(data_args, datasets['test'], qa_tokenizer, rewrites, use_token_type_ids = "xlnet" in model_args.qa_model_name_or_path, use_history_for_qa=training_args.use_history_for_qa, is_coqa=is_coqa)
        os.makedirs(f"{training_args.output_dir}/{training_args.exp}", exist_ok=True)

        qa_training_args = copy.deepcopy(training_args)
        qa_training_args.report_to = ['tensorboard']
        trainer = QuestionAnsweringTrainer(
                    model=qa_model,
                    args=qa_training_args,
                    tokenizer=qa_tokenizer,
                    data_collator=default_data_collator if is_coqa else DataCollatorWithIgnoreComponents(),
                    post_process_function=post_processing_function_eval,
                    compute_metrics=compute_metrics,
                )
        print("*** TEST | Run Trainer.Predict ***")
        test_output = trainer.predict(
                    test_dataset=test_dataset,
                    test_examples=datasets["test"],
                    metric_key_prefix="eval",
                )
        results = test_output.metrics
        print(f"F1:{results['eval_f1']} EM: {results['eval_exact_match']}")
        print()

        output_eval_file = os.path.join(training_args.output_dir, training_args.exp, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                print("***** Evaluation results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        return results


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, REINFORCETrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)