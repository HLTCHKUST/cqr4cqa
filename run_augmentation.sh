# CoQA augmentation with QReCC
CUDA_VISIBLE_DEVICES=0 python run_data_augmentation.py\
 --do_train \
 --epochs 10 \
 --ppo_epochs 1 \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 2 \
 --per_device_eval_batch_size 16 \
 --repetition_penalty 1.0 \
 --num_beams 1 \
 --sampling True \
 --p 0.95 \
 --k 10 \
 --num_return_sequences 10 \
 --xent_init_length 0 \
 --xent_min_len 0 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-qrecc \
 --ppo_model_name_or_path save/gpt2-qrecc \
 --lr 1e-7 \
 --output_dir save \
 --exp coqa-augment-qrecc


# QuAC augmentation with QReCC
CUDA_VISIBLE_DEVICES=0 python run_data_augmentation.py \
 --do_train \
 --epochs 10 \
 --ppo_epochs 1\
 --dataset_name src/data_utils/quac.py \
 --dataset_config_name quac_ppo \
 --length 50 \
 --gen_max_seq_length 150 \
 --batch_size 2 \
 --per_device_eval_batch_size 16 \
 --repetition_penalty 1.0 \
 --num_beams 1 \
 --sampling True \
 --p 0.95 \
 --k 10 \
 --num_return_sequences 10 \
 --xent_init_length 0 \
 --xent_min_len 0 \
 --qa_tokenizer_name save/roberta-base-quac \
 --qa_model_name_or_path save/roberta-base-quac \
 --ppo_tokenizer_name save/gpt2-qrecc \
 --ppo_model_name_or_path save/gpt2-qrecc \
 --lr 1e-7 \
 --optimizer Adam \
 --output_dir save\
 --exp quac-augment-qrecc