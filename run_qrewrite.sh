DATASET=$1
OUTPUT_DIR=$2

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$3 run_qrewrite.py \
--dataset_name $DATASET \
--output_dir $OUTPUT_DIR \
--num_train_epochs 10 \
--model_name_or_path gpt2 \
--early_stopping_patience 3 \
--learning_rate 3e-5 \
--adam_epsilon 1e-06 \
--gradient_accumulation_steps 16 \
--do_train \
--do_eval \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--overwrite_output_dir \
--adam_eps 1e-06 \
--max_seq_length 256 \
--warmup_steps 500 \
--evaluation_strategy epoch \
--load_best_model_at_end \
--save_total_limit 5 \
--logging_steps 50 \
--sharded_ddp \
--overwrite_cache 