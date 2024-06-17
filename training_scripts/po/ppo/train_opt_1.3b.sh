#!/bin/bash

set -e
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATA_PATH="./datasets/Dahoas/full-hh-rlhf"
BASE_PATH="result"
ACTOR_MODEL_PATH="${BASE_PATH}/step1_sft-facebook_opt-1.3b-full_hh_rlhf-2023-12-23-10-02-24-1234"
CRITIC_MODEL_PATH="${BASE_PATH}/step2_reward-facebook_opt-1.3b-full_hh_rlhf-2023-12-23-17-36-42-1235"
ACTOR_ZERO_STAGE=2
CRITIC_ZERO_STAGE=2
REFERENCE_ZERO_STAGE=0
REWARD_ZERO_STAGE=0
LOG_PATH="log"
SEED=1234

TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT="$LOG_PATH/step3_ppo-opt_1.3b-full_hh_rlhf-$TIME_STEP-$SEED"

mkdir -p $OUTPUT

ACTOR_LR=1e-6
CRITIC_LR=1e-6

deepspeed --master_port 12346  step3_ppo_finetuning/main.py \
   --algo "ppo" \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/rlhf_consistency/opt" \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${ACTOR_LR} \
   --critic_learning_rate ${CRITIC_LR} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload \
   --offload_reference_model \
   --offload_reward_model \
   --num_warmup_steps 0 \
   --kl_ctl 0.1 \
   --alpha_ctl 0.0 \
   --deepspeed \
   --dtype bf16 \
   --seed $SEED \
   --data_seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_answers \
   --save_answers \
   --eval_samples 500 \
   &> $OUTPUT/training.log
