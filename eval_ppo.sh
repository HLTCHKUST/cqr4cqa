# QReCC + CoQA
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_eval \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 16 \
 --per_device_eval_batch_size 16 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-qrecc-coqa \
 --ppo_model_name_or_path save/gpt2-qrecc-coqa \
 --output_dir save \
 --exp gpt2-qrecc-coqa \
 --repetition_penalty 1.1 \
 --num_beams 5 \


# CANARD + CoQA
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_eval \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 16 \
 --per_device_eval_batch_size 16 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-canard-coqa \
 --ppo_model_name_or_path save/gpt2-canard-coqa \
 --output_dir save \
 --exp gpt2-canard-coqa \
 --repetition_penalty 1.1 \
 --num_beams 5 \


# QReCC + QuAC
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_eval \
 --dataset_name src/data_utils/quac.py \
 --dataset_config_name quac_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 8 \
 --per_device_eval_batch_size 8 \
 --qa_tokenizer_name save/roberta-base-quac \
 --qa_model_name_or_path save/roberta-base-quac \
 --ppo_tokenizer_name save/gpt2-qrecc-quac \
 --ppo_model_name_or_path save/gpt2-qrecc-quac \
 --output_dir save \
 --exp gpt2-qrecc-quac \
 --repetition_penalty 1.1 \
 --num_beams 5 \


# CANARD + QuAC
CUDA_VISIBLE_DEVICES=0 python run_ppo.py \
 --do_eval \
 --dataset_name src/data_utils/quac.py \
 --dataset_config_name quac_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 8 \
 --per_device_eval_batch_size 8 \
 --qa_tokenizer_name save/roberta-base-quac \
 --qa_model_name_or_path save/roberta-base-quac \
 --ppo_tokenizer_name save/gpt2-canard-quac \
 --ppo_model_name_or_path save/gpt2-canard-quac \
 --output_dir save \
 --exp gpt2-canard-quac \
 --repetition_penalty 1.1 \
 --num_beams 5 \