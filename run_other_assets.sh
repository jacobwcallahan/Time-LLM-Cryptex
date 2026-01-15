#!/usr/bin/env bash

MODEL_ID="hourly_1week_to_1day_50pct_20251104_172640"
DATA_DIR="./dataset/other_assets"
SAVE_DIR="${DATA_DIR}/infs"
LLM_MODEL="LLAMA3.1"

mkdir -p "${SAVE_DIR}"

run_asset () {
  ASSET="$1"

  echo "Running inference for ${ASSET}"

  python3 run_inference.py \
    --model_id "${MODEL_ID}" \
    --data_path "${DATA_DIR}/${ASSET}.csv" \
    --save_path "${SAVE_DIR}" \
    --llm_model "${LLM_MODEL}"

  mv "${SAVE_DIR}/inference.csv" "${SAVE_DIR}/${ASSET}-inf.csv"
}

run_asset "BTC-USD"

# run_asset "AAPL"
# run_asset "ADA-USD"
# run_asset "BCH-USD"
# run_asset "DOGE-USD"
# run_asset "ETH-USD"
# run_asset "EURUSD-X"
# run_asset "JPY-X"
# run_asset "LTC-USD"
# run_asset "NVDA"
# run_asset "SOL-USD"
# run_asset "TSLA"
# run_asset "XRP-USD"

echo "All inference jobs completed."
