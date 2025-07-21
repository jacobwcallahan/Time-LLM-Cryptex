#!/bin/bash

# --- User-Configurable Parameters ---
LLM_MODEL="LLAMA"
LLM_LAYERS=2
GRANULARITY="daily" # Choices: hourly, minute, daily
TASK_NAME="short_term_forecast" # Choices: long_term_forecast, short_term_forecast
FEATURES="S" # Choices: M, MS, S
SEQ_LEN=14
PRED_LEN=6
PATCH_LEN=1
STRIDE=1
NUM_TOKENS=100
LLM_DIM=4096
LOSS="MSE"
METRIC="MAE"

# --- Dynamic Parameter Logic ---
# 1. Generate Model ID
MODEL_ID="${LLM_MODEL}_L${LLM_LAYERS}_${GRANULARITY}_${FEATURES}_seq${SEQ_LEN}_pred${PRED_LEN}_p${PATCH_LEN}_s${STRIDE}_v${NUM_TOKENS}"

# 2. Set Data Path based on Granularity
case $GRANULARITY in
  "hourly") DATA_PATH="candlesticks-h.csv" ;;
  "minute") DATA_PATH="candlesticks-Min.csv" ;;
  "daily") DATA_PATH="candlesticks-D.csv" ;;
esac

# 3. Set Label Length
LABEL_LEN=$((SEQ_LEN / 2))

# --- Static Configuration ---
MASTER_PORT="29500"
NUM_PROCESS="4"
D_MODEL="32"
D_FF="128"
TRAIN_EPOCHS="10"
LEARNING_RATE="0.01"
BATCH_SIZE="24"
ROOT_PATH="./dataset/cryptex/"

# --- Execution ---
echo "Launching experiment with Model ID: $MODEL_ID"
echo "Data Path: $DATA_PATH"
echo "Label Length: $LABEL_LEN"

accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes $NUM_PROCESS \
    --main_process_port $MASTER_PORT \
    run_main.py \
    --task_name $TASK_NAME \
    --is_training 1 \
    --model_id "$MODEL_ID" \
    --model_comment "TimeLLM-Cryptex" \
    --llm_model $LLM_MODEL \
    --data "CRYPTEX" \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --features $FEATURES \
    --target "close" \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --patch_len $PATCH_LEN \
    --stride $STRIDE \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --factor 3 \
    --itr 1 \
    --train_epochs $TRAIN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --llm_layers $LLM_LAYERS \
    --model "TimeLLM" \
    --num_tokens $NUM_TOKENS \
    --llm_dim $LLM_DIM \
    --loss $LOSS \
    --metric $METRIC