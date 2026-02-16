from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainConfig:
    """Dataclass holding all CLI arguments for run_main.py training script.

    All fields have defaults. You can create instances manually by passing
    only the values you want to override:

        config = TrainConfig(model_id="my_run", batch_size=64, learning_rate=0.001)
        config = TrainConfig(seq_len=128, pred_len=48)  # use defaults for the rest
    """

    # Basic experiment config
    model_id: str = "test"
    seed: int = 2021

    # Data loader arguments
    data: str = "CRYPTEX"
    root_path: str = "./dataset"
    data_path: str = "candlesticks-D.csv"
    features: str = "MS"
    target: str = "close"
    checkpoints: str = "./checkpoints/"

    # Forecasting task arguments
    seq_len: int = 96
    pred_len: int = 96

    # Model architecture arguments
    enc_in: int = 7
    d_model: int = 16
    n_heads: int = 8
    d_ff: int = 32
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    llm_model: str = "LLAMA"

    # Optimization and training arguments
    num_workers: int = 10
    train_epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 8
    patience: int = 10
    learning_rate: float = 0.0001
    loss: str = "MSE"
    metric: str = "MAE"
    lradj: str = "type1"
    pct_start: float = 0.2
    use_amp: bool = False
    llm_layers: int = 6
    percent: int = 100
    num_tokens: int = 1000
    enable_mlflow: bool = True
    experiment_name: Optional[str] = None

    @classmethod
    def from_namespace(cls, ns) -> "TrainConfig":
        """Build TrainConfig from argparse.Namespace (e.g. parse_args() output)."""
        return cls(**{k: v for k, v in vars(ns).items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Build TrainConfig from a dict (e.g. from YAML/JSON config). Ignores unknown keys."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def parse(cls) -> "TrainConfig":
        """Parse CLI arguments and return a TrainConfig."""
        import argparse

        parser = argparse.ArgumentParser(description='Time-LLM Training Script')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='Unique identifier for this training run (used for logging, checkpointing, and experiment tracking).')
        parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducibility across runs (affects data shuffling, weight initialization, etc.).')

        # Data loader arguments
        parser.add_argument('--data', type=str, required=True, default='CRYPTEX', help='Dataset name/type to use. For this project, should be "CRYPTEX".')
        parser.add_argument('--root_path', type=str, default='./dataset', help='Root directory where all data files are stored.')
        parser.add_argument('--data_path', type=str, default='candlesticks-D.csv', help='Filename of the main data CSV to use for training/validation/testing.')
        parser.add_argument('--features', type=str, default='MS', help='Forecasting task type: "M": multivariate predict multivariate, "S": univariate predict univariate, "MS": multivariate predict univariate')
        parser.add_argument('--target', type=str, default='close', help='Name of the target feature/column to forecast (used for S or MS tasks).')

        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory where model checkpoints and temporary files will be saved during training.')

        # Forecasting task arguments
        parser.add_argument('--seq_len', type=int, default=96, help='Length of the input sequence (number of time steps fed into the model).')
        parser.add_argument('--pred_len', type=int, default=96, help='Length of the prediction horizon (number of future time steps to forecast).')

        # Model architecture arguments
        parser.add_argument('--enc_in', type=int, default=7, help='Number of input features for RevIN (if affine=True).')
        parser.add_argument('--d_model', type=int, default=16, help='Dimensionality of the patch embeddings after the PatchEmbedder.')
        parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads in the Reprogramming Layer (for multi-head attention layers).')
        parser.add_argument('--d_ff', type=int, default=32, help='Dimensionality of the feedforward network at the output layer (hard-sliced).')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate applied throughout the model to prevent overfitting.')

        parser.add_argument('--patch_len', type=int, default=16, help='Patch length for patch-based models (number of time steps per patch).')
        parser.add_argument('--stride', type=int, default=8, help='Stride for patch-based models (step size between patches).')
        parser.add_argument('--llm_model', type=str, default='LLAMA', help='Name of the LLM model (for experiment tracking and logging purposes).')

        # Optimization and training arguments
        parser.add_argument('--num_workers', type=int, default=10, help='Number of worker processes for data loading (higher values may speed up data loading).')
        parser.add_argument('--train_epochs', type=int, default=10, help='Total number of training epochs (full passes through the training dataset).')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (number of samples per training step).')
        parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for evaluation/validation (number of samples per evaluation step).')
        parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before early stopping (prevents overfitting).')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate for the optimizer.')
        parser.add_argument('--loss', type=str, default='MSE', help='Loss function to use for training (e.g., "MSE" for mean squared error).')
        parser.add_argument('--metric', type=str, default='MAE', help='Evaluation metric to use (e.g., "MAE" for mean absolute error).')
        parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy: "type1", "COS", or "TST".')
        parser.add_argument('--pct_start', type=float, default=0.2, help='Percentage of the OneCycleLR schedule spent increasing the learning rate (used if OneCycleLR is selected).')
        parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP) training for faster and more memory-efficient training on supported hardware.')
        parser.add_argument('--llm_layers', type=int, default=6, help='Number of LLM layers to use (if applicable to the model).')
        parser.add_argument('--percent', type=int, default=100, help='Percentage of the dataset to use for training (useful for quick experiments or ablation studies).')
        parser.add_argument('--num_tokens', type=int, default=1000, help='Number of tokens for the mapping layer (controls tokenization granularity).')
        parser.add_argument('--enable_mlflow', action='store_true', default=True, help='Enable MLflow experiment tracking and logging (recommended: keep enabled).')
        parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name to use for MLflow experiment tracking and logging.')
        ns = parser.parse_args()
        return cls.from_namespace(ns)
