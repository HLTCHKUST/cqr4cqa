python -m torch.distributed.launch --nproc_per_node=1 --master_port=10001 run_convqa.py \
 --dataset_name  'src/data_utils/quac.py'\
 --dataset_config_name quac_rc \
 --model_name_or_path roberta-base \
 --do_train \
 --do_eval \
 --early_stop \
 --early_stopping_patience 4 \
 --logging_steps 100 \
 --save_steps 500 \
 --learning_rate 3e-5  \
 --num_train_epochs 10\
 --max_seq_length 512 \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_prefix roberta-base \
 --cache_dir cache \
 --output_dir save/roberta-base-quac \
 --overwrite_output_dir \
 --preprocessing_num_workers 10 \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 6 \
 --gradient_accumulation_steps 20  \
 --evaluation_strategy epoch \
 --load_best_model_at_end \
 --metric_for_best_model f1 \
 --warmup_steps 0 \
 --weight_decay 0.01 \
 --sharded_ddp 