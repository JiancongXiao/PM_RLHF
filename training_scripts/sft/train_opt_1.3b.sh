#!/bin/bash
set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team

ZERO_STAGE=2
DATA_PATH="./datasets/Dahoas/full-hh-rlhf" 
MODEL_NAME="facebook/opt-1.3b"
LOG_PATH="log"
SEED=1234


TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT="${LOG_PATH}/step1_sft-${MODEL_NAME/'/'/_}-full_hh_rlhf-$TIME_STEP-$SEED"

mkdir -p $OUTPUT


deepspeed  --master_port 12345 step1_supervised_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/rlhf_consistency/opt" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 30 \
   --per_device_eval_batch_size 30 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed $SEED \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --dtype bf16 \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
