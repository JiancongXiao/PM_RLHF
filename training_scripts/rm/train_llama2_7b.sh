#!/bin/sh
#SBATCH -J llama3
#SBATCH -o log/%j.out
#SBATCH -A L00120230003
#SBATCH -p p-A800
#SBATCH -w pgpu26
#SBATCH --gres=gpu:4
#SBATCH -n 24

set -e 
set -x

# module load cuda11.8/toolkit/11.8.0
# module load gcc/11.2.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

# source /mntcephfs/lab_data/liziniu/anaconda3/etc/profile.d/conda.sh
# conda activate rlhf

# export CUDA_HOME=/mntcephfs/lab_data/liziniu/cuda-12.1
# export PATH=$PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mntcephfs/lab_data/liziniu/cuda-12.1/lib64

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/220049033/huggingface"

# DeepSpeed Team
BASE_PATH="/220049033"
DATA_PATH="${BASE_PATH}/datasets/tl-dr"
MODEL_NAME="meta-llama/Llama-3.2-1B"
ZERO_STAGE=2

LOG_PATH="./log"
SEED=1234

TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT="${LOG_PATH}/step2_reward-Llama3_1b-tl_dr-$TIME_STEP-$SEED"

mkdir -p $OUTPUT

deepspeed --master_port 12349 step2_reward_model_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_output_path "${BASE_PATH}/tmp/data_files/rlhf_consistency/llama3" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 36 \
   --per_device_eval_batch_size 36 \
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
   --dtype bf16 \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
  2>&1 | tee $OUTPUT/training.log