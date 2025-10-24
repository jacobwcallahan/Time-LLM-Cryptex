#!/bin/bash
# Download inference CSV files from remote server
# Usage: ./download_inference_csvs.sh [server_name] [local_dir]

# Configuration
REMOTE_HOST="${1:-gpu3}"
LOCAL_DIR="${2:-./inference_results}"
REMOTE_DIR="~/Time-LLM-Cryptex/inference_results/"

echo "========================================"
echo "Downloading inference CSVs from $REMOTE_HOST"
echo "========================================"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Check if remote directory exists
echo "Checking remote directory..."
if ! ssh "$REMOTE_HOST" "test -d ${REMOTE_DIR}"; then
    echo "ERROR: Remote directory ${REMOTE_DIR} does not exist on $REMOTE_HOST"
    echo "Make sure inference has been run on the remote server first."
    exit 1
fi

# List remote CSV files
echo "Remote CSV files found:"
ssh "$REMOTE_HOST" "ls -la ${REMOTE_DIR}*.csv 2>/dev/null | wc -l" | xargs -I {} echo "Found {} CSV files"

# Download all CSV files
echo ""
echo "Downloading CSV files..."
rsync -avzP \
  --include="*.csv" \
  --exclude="*" \
  "${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR/"

# Check if any files were downloaded
DOWNLOADED_COUNT=$(find "$LOCAL_DIR" -name "*.csv" -type f | wc -l)

if [ "$DOWNLOADED_COUNT" -eq 0 ]; then
    echo ""
    echo "WARNING: No CSV files were downloaded!"
    echo "This could mean:"
    echo "  1. No inference has been run on the remote server"
    echo "  2. The remote directory path is incorrect"
    echo "  3. There are no CSV files in the remote directory"
    echo ""
    echo "To check manually:"
    echo "  ssh $REMOTE_HOST 'ls -la ${REMOTE_DIR}'"
    exit 1
fi

echo ""
echo "========================================"
echo "Download Summary"
echo "========================================"
echo "Downloaded $DOWNLOADED_COUNT CSV files to: $LOCAL_DIR"
echo ""
echo "Files downloaded:"
ls -la "$LOCAL_DIR"/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes)"}'

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo "1. Run batch backtesting:"
echo "   python run_batch_backtest.py --inference_dir $LOCAL_DIR"
echo ""
echo "2. Or run individual backtests:"
echo "   python backtesting/backtest.py --data $LOCAL_DIR/filename.csv"
echo ""
echo "3. Check file contents:"
echo "   head $LOCAL_DIR/filename.csv"
echo ""

# Optional: Download backtest results if they exist
echo "Checking for existing backtest results..."
if ssh "$REMOTE_HOST" "test -d ${REMOTE_DIR}backtest_results"; then
    echo "Found backtest results on remote server."
    read -p "Download backtest results too? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading backtest results..."
        rsync -avzP \
          "${REMOTE_HOST}:${REMOTE_DIR}backtest_results/" "$LOCAL_DIR/backtest_results/"
        echo "Backtest results downloaded to: $LOCAL_DIR/backtest_results/"
    fi
fi

echo ""
echo "Download complete!"
echo "========================================"
