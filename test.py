"""
Smoke test for run_main.py training pipeline.
Uses TrainConfig with minimal settings for a quick run (1 epoch, small data subset).
"""
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infra.TrainConfig import TrainConfig
from run_main import main


def test_run_main_training():
    """Run a minimal training pass to verify the pipeline works."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Use dataset/candles/candlesticks-D.csv if it exists, else dataset/candlesticks-D.csv
    candles_path = os.path.join(project_root, "dataset", "candles", "candlesticks-D.csv")
    default_path = os.path.join(project_root, "dataset", "candlesticks-D.csv")
    if os.path.exists(candles_path):
        root_path = os.path.join(project_root, "dataset", "candles")
        data_path = "candlesticks-D.csv"
    elif os.path.exists(default_path):
        root_path = os.path.join(project_root, "dataset")
        data_path = "candlesticks-D.csv"
    else:
        raise FileNotFoundError(
            f"No dataset found. Expected candlesticks-D.csv in dataset/ or dataset/candles/"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainConfig(
            model_id="test_run",
            root_path=root_path,
            data_path=data_path,
            train_epochs=1,
            batch_size=8,
            eval_batch_size=4,
            enable_mlflow=True,
        )
        main(config)


if __name__ == "__main__":
    test_run_main_training()
    print("Test passed: run_main training completed successfully.")
