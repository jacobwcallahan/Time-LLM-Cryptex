#!/bin/bash
# Sync local code to remote server (exact mirror)
# Usage: ./sync_to_server.sh [server_name]

# Configuration
REMOTE_HOST="${1:-gpu3}"
LOCAL_DIR="/home/jchmura/github/workspaces/Time-LLM-Cryptex/"
REMOTE_DIR="~/Time-LLM-Cryptex/"

echo "========================================"
echo "Syncing to $REMOTE_HOST"
echo "========================================"

# Rsync with exact mirroring but preserve CSV files and previously ignored content
rsync -avzP \
  --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.pyd' \
  --exclude '.pytest_cache/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'checkpoints/' \
  --exclude 'mlruns/' \
  --exclude '*.egg-info/' \
  --exclude 'dist/' \
  --exclude 'build/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '.DS_Store' \
  --exclude 'dataset/cryptex/daily/cache/*.csv' \
  --exclude 'inference_results/*.csv' \
  --exclude 'inference_results/backtest_results/' \
  --exclude '*.log' \
  "$LOCAL_DIR" "${REMOTE_HOST}:${REMOTE_DIR}"

echo ""
echo "========================================"
echo "Cleaning Python cache on server..."
echo "========================================"

# Clean Python cache on remote
ssh "$REMOTE_HOST" "cd ${REMOTE_DIR} && \
  find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
  find . -type f -name '*.pyc' -delete; \
  find . -type f -name '*.pyo' -delete; \
  echo 'Cache cleaned successfully'"

echo ""
echo "========================================"
echo "Sync complete!"
echo "========================================"
echo "To verify: ssh $REMOTE_HOST 'cd ${REMOTE_DIR} && git status'"
echo ""