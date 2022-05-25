# QReCC + CoQA
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_train \
 --epochs 6 \
 --ppo_epochs 1 \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 16 \
 --per_device_eval_batch_size 16 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-qrecc \
 --ppo_model_name_or_path save/gpt2-qrecc \
 --lr 1e-7 \
 --output_dir save \
 --exp gpt2-qrecc-coqa \
 --repetition_penalty 1.1 \
 --num_beams 5 \
 --vf_coef 0.5 \
 --xent_init_length 3 \
 --mixer_steps 12000 \
 --xent_min_length 1


# CANARD + CoQA
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_train \
 --epochs 6 \
 --ppo_epochs 1 \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 16 \
 --per_device_eval_batch_size 16 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-canard \
 --ppo_model_name_or_path save/gpt2-canard \
 --lr 1e-7 \
 --output_dir save \
 --exp gpt2-canard-coqa \
 --repetition_penalty 1.1 \
 --num_beams 5 \
 --vf_coef 0.5 \
 --xent_init_length 3 \
 --mixer_steps 12000 \
 --xent_min_length 1


# QReCC + QuAC
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_train \
 --epochs 6 \
 --ppo_epochs 1 \
 --dataset_name src/data_utils/quac.py \
 --dataset_config_name quac_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 8 \
 --per_device_eval_batch_size 8 \
 --qa_tokenizer_name save/roberta-base-quac \
 --qa_model_name_or_path save/roberta-base-quac \
 --ppo_tokenizer_name save/gpt2-qrecc \
 --ppo_model_name_or_path save/gpt2-qrecc \
 --lr 1e-7 \
 --output_dir save \
 --exp gpt2-qrecc-quac \
 --repetition_penalty 1.1 \
 --num_beams 5 \
 --vf_coef 0.5 \
 --xent_init_length 3 \
 --mixer_steps 12000 \
 --xent_min_length 1


# CANARD + QuAC
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_train \
 --epochs 6 \
 --ppo_epochs 1 \
 --dataset_name src/data_utils/quac.py \
 --dataset_config_name quac_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 8 \
 --per_device_eval_batch_size 8 \
 --qa_tokenizer_name save/roberta-base-quac \
 --qa_model_name_or_path save/roberta-base-quac \
 --ppo_tokenizer_name save/gpt2-canard \
 --ppo_model_name_or_path save/gpt2-canard \
 --lr 1e-7 \
 --output_dir save \
 --exp gpt2-canard-quac \
 --repetition_penalty 1.1 \
 --num_beams 5 \
 --vf_coef 0.5 \
 --xent_init_length 3 \
 --mixer_steps 12000 \
 --xent_min_length 1