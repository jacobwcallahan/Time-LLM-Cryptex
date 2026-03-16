"""
Single-run training for Time-LLM-Cryptex.
Trains one model with fixed hyperparameters (no Optuna HPO).
Excludes inference and backtest options.
"""

import argparse
import os
import shutil
import uuid
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Single-run training (no HPO)")
    # HPO args (excluding inference, backtest)
    parser.add_argument("--gpu", type=str, default="1", help="GPU to use")
    parser.add_argument("--study_name", type=str, default="", help="Study name (for logging)")
    parser.add_argument("--granularity", type=str, default="daily", choices=["daily", "hourly", "weekly", "minute"])
    parser.add_argument("--start", type=str, default=None, help="Training start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--data_path", type=str, default=None, help="Custom data path")
    parser.add_argument("--returns", action="store_true", help="Train on returns")
    parser.add_argument("--volatility", action="store_true", help="Use volatility target")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--aggregate", type=int, default=1)
    parser.add_argument("--no_inf_aggregate", action="store_true")
    parser.add_argument("--log_all_metrics", action="store_true")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID for MLflow run name")
    parser.add_argument("--prompt_file", type=str, default="custom_prompt", help="Prompt filename without .txt (loads from dataset/prompt_bank/{prompt_file}.txt)")

    # Single model params (categorical)
    parser.add_argument("--features", type=str, default="S", choices=["S", "MS", "M"])
    parser.add_argument("--seq_len", type=int, default=180)
    parser.add_argument("--pred_len", type=int, default=14)
    parser.add_argument("--num_tokens", type=int, default=500)
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "MADL", "GMADL", "MADLSTE", "SHARPE"])
    parser.add_argument("--lradj", type=str, default="TST", choices=["type1", "type2", "type3", "PEMS", "TST", "COS", "constant"])
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_len", type=int, default=14)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)

    # Single model params (int/float)
    parser.add_argument("--llm_layers", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pct_start", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    return parser.parse_args()


def main():
    cli = parse_args()

    # Set GPU before any CUDA/torch imports
    if cli.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

    # Import after CUDA_VISIBLE_DEVICES is set
    from hpo_core.DataManager import DataManager
    from hpo_core.WorkDir import WorkDir
    from hpo_core.HpoArgs import HpoArgs
    from hpo_core.PipelineRunner import PipelineRunner

    # Build HpoArgs for WorkDir/DataManager (no inference/backtest)
    hpo_overrides = {
        "gpu": cli.gpu,
        "study_name": cli.study_name,
        "granularity": cli.granularity,
        "start": cli.start,
        "end": cli.end,
        "data_path": cli.data_path,
        "returns": cli.returns,
        "volatility": cli.volatility,
        "experiment_name": cli.experiment_name,
        "aggregate": cli.aggregate,
        "no_inf_aggregate": cli.no_inf_aggregate,
        "log_all_metrics": cli.log_all_metrics,
        "backtest": False,
        "inf_start": None,
        "inf_end": None,
        "trials": 1,
        "yaml_file": "optuna_vars.yaml",
    }
    args = HpoArgs(parse_cli=False, **hpo_overrides)

    work_dir = WorkDir(args)
    work_dir.create_work_dir()

    print("Prepping data...")
    data_manager = DataManager(work_dir=work_dir)
    data_manager.prepare_train_data(
        start=args.start, end=args.end,
        aggregate=args.aggregate, returns=args.returns
    )
    print("Data prepped. Starting training...")

    model_id = cli.model_id
    if not model_id:
        model_id = f"single_{uuid.uuid4().hex[:8]}_{args.granularity}_seq{cli.seq_len}_pred{cli.pred_len}"

    config = {
        "features": cli.features,
        "seq_len": cli.seq_len,
        "pred_len": cli.pred_len,
        "num_tokens": cli.num_tokens,
        "loss": cli.loss,
        "lradj": cli.lradj,
        "n_heads": cli.n_heads,
        "d_ff": cli.d_ff,
        "batch_size": cli.batch_size,
        "patch_len": cli.patch_len,
        "stride": cli.stride,
        "epochs": cli.epochs,
        "llm_layers": cli.llm_layers,
        "d_model": cli.d_model,
        "dropout": cli.dropout,
        "pct_start": cli.pct_start,
        "learning_rate": cli.learning_rate,
        "llm_model": "LLAMA3.1",
        "metric": "MSE",
        "target": "volatility" if args.volatility else ("returns" if args.returns else "close"),
        "experiment_name": args.experiment_name,
        "model_id": model_id,
        "data_path": str(work_dir.get_train_data_path()),
        "data": cli.prompt_file,
    }

    pipeline_runner = PipelineRunner(work_dir)
    pipeline_runner.run_training(config)

    print("\n--- Single Run Training Finished ---")
    if Path("temp").exists():
        shutil.rmtree("temp")


if __name__ == "__main__":
    main()
