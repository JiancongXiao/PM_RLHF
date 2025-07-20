#!/bin/bash

set -e 
set -x

export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=2
DATA_PATH="/home/datasets/Dahoas/full-hh-rlhf"
MODEL_NAME="facebook/opt-1.3b"
LOG_PATH="../log"
SEED=1235

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="${LOG_PATH}/step2_reward-${MODEL_NAME/'/'/_}-full_hh_rlhf-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT

# --offload \

deepspeed --master_port 12349 main.py \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/rlhf_consistency/opt" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 24 \
   --per_device_eval_batch_size 24 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed $SEED \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
