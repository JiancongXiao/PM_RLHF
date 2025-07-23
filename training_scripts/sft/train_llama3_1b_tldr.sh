set -e 
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

# cuda-12.1

BASE_PATH="./home"
DATA_PATH="${BASE_PATH}/datasets/tl-dr"
MODEL_NAME="meta-llama/Llama-3.2-1B"
ZERO_STAGE=2

LOG_PATH="./log"
SEED=1234

TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
OUTPUT="${LOG_PATH}/step1_sft-Llama3_1b-tl_dr-$TIME_STEP-$SEED"

mkdir -p $OUTPUT

deepspeed  --master_port 12345 step1_supervised_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_output_path "${BASE_PATH}/tmp/data_files/rlhf_consistency/llama3" \
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
   --save_model \
  2>&1 | tee $OUTPUT/training.log
