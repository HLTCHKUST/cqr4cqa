# QReCC + CoQA + REINFORCE
# change dataset_name, dataset_config_name, and model path for your desired combination
CUDA_VISIBLE_DEVICES=0 python run_reinforce.py\
 --do_train \
 --epochs 5 \
 --dataset_name src/data_utils/coqa.py \
 --dataset_config_name coqa_ppo \
 --length 50\
 --gen_max_seq_length 150 \
 --batch_size 32 \
 --per_device_eval_batch_size 32 \
 --qa_tokenizer_name save/roberta-base-coqa \
 --qa_model_name_or_path save/roberta-base-coqa \
 --ppo_tokenizer_name save/gpt2-qrecc \
 --ppo_model_name_or_path save/gpt2-qrecc \
 --lr 1e-6 \
 --output_dir save \
 --exp gpt2-qrecc-coqa-reinforce