python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 run_convqa.py \
 --dataset_name  'src/data_utils/coqa.py'\
 --dataset_config_name coqa_rc \
 --model_name_or_path roberta-base \
 --eval_coqa \
 --do_train \
 --do_eval \
 --early_stop \
 --early_stopping_patience 3 \
 --logging_steps 100 \
 --save_steps 500 \
 --learning_rate 3e-5  \
 --num_train_epochs 10 \
 --max_seq_length 512 \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir cache\
 --output_dir save/roberta-base-coqa \
 --overwrite_output_dir  \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 6 \
 --gradient_accumulation_steps 10  \
 --evaluation_strategy epoch \
 --load_best_model_at_end \
 --metric_for_best_model f1 \
 --warmup_steps 1000 \
 --weight_decay 0.01 \
 --sharded_ddp 