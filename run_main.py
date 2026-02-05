import argparse
import math
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import socket
from contextlib import nullcontext
from utils.metrics import get_loss_function, get_metric_function
from models import TimeLLM
from utils.data_provider import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali
import shutil

import mlflow
import mlflow.pytorch

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    """
    Parse command-line arguments for the TimeLLM training script.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Time-LLM Training Script')

    # Basic experiment config
    parser.add_argument('--model_id', type=str, required=True, default='test', help='Unique identifier for this training run (used for logging, checkpointing, and experiment tracking).')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducibility across runs (affects data shuffling, weight initialization, etc.).')

    # Data loader arguments
    parser.add_argument('--data', type=str, required=False, default=None, help='Dataset name/type to use. Auto-detected from data_path if contains "features_", otherwise defaults to "CRYPTEX".')
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
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Fraction of total steps for linear warmup (used with lradj WARM_COS).')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine decay (used with lradj WARM_COS).')
    parser.add_argument('--plateau_patience', type=int, default=2, help='Epochs without vali improvement before plateau restart (lradj PLATEAU_RESTART).')
    parser.add_argument('--plateau_delta', type=float, default=0.0, help='Min improvement in vali loss to count as improvement (lradj PLATEAU_RESTART).')
    parser.add_argument('--peak_decay', type=float, default=0.8, help='Factor to multiply peak LR by on plateau-triggered restart (lradj PLATEAU_RESTART).')
    parser.add_argument('--restart_T0_epochs', type=float, default=1.0, help='First cosine cycle length in epochs (lradj PLATEAU_RESTART).')
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP) training for faster and more memory-efficient training on supported hardware.')
    parser.add_argument('--llm_layers', type=int, default=6, help='Number of LLM layers to use (if applicable to the model).')
    parser.add_argument('--percent', type=int, default=100, help='Percentage of the dataset to use for training (useful for quick experiments or ablation studies).')
    parser.add_argument('--num_tokens', type=int, default=1000, help='Number of tokens for the mapping layer (controls tokenization granularity).')
    parser.add_argument('--enable_mlflow', action='store_true', default=True, help='Enable MLflow experiment tracking and logging (recommended: keep enabled).')
    parser.add_argument('--gradient_clip_norm', type=float, default=None, help='Gradient clipping norm value (set to disable by default).')

    return parser.parse_args()

def detect_feature_set_from_path(data_path: str) -> str:
    """
    Detect feature set name from data_path and return appropriate prompt file name.
    
    Args:
        data_path: Path to data CSV file
    
    Returns:
        Prompt file name (without .txt extension) for --data argument
    """
    # Check if this is a feature-engineered dataset
    if 'features_' in data_path:
        # Extract feature set name from path like "candlesticks-D_features_momentum.csv"
        parts = data_path.split('_features_')
        if len(parts) == 2:
            feature_set = parts[1].replace('.csv', '')
            return f'CRYPTEX_features_{feature_set}'
    
    # Default to standard CRYPTEX prompt
    return 'CRYPTEX'


def run_training(args, accelerator):
    """
    Run training with automatic feature set detection.
    Main training/validation/testing logic for TimeLLM.
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        accelerator: Accelerator object for distributed/mixed precision training
    """
    enable_mlflow = args.enable_mlflow

    # Only the main process should create an MLflow run and log artifacts/metrics
    if enable_mlflow and accelerator.is_main_process:
        mlflow.set_experiment(args.llm_model)
        run_context = mlflow.start_run(run_name=args.model_id)
        hostname = socket.gethostname()
        mlflow.set_tag("hostname", hostname)
    else:
        run_context = nullcontext()

    with run_context:
        # Log all arguments to MLflow if enabled and on main process
        if enable_mlflow and accelerator.is_main_process:
            mlflow.log_params(vars(args))

        # Load data for training, validation, and testing
        train_data, train_loader = data_provider(args, 'train')  # train_data: Dataset, train_loader: DataLoader
        vali_data, vali_loader = data_provider(args, 'val')      # vali_data: Dataset, vali_loader: DataLoader
        test_data, test_loader = data_provider(args, 'test')     # test_data: Dataset, test_loader: DataLoader

        # Initialize the TimeLLM model
        model = TimeLLM.Model(args).float()  # Model expects input: [batch, seq_len, num_features]
        temp_checkpoint_path = os.path.join(args.checkpoints, args.model_id)

        if accelerator.is_local_main_process:
            os.makedirs(temp_checkpoint_path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        # Set up optimizer and learning rate scheduler
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        elif args.lradj == 'constant':
            # Constant learning rate - no adjustment
            scheduler = torch.optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lambda epoch: 1.0)
        elif args.lradj == 'type3':
            # type3 uses manual epoch-based adjustment, so use LambdaLR (not OneCycleLR)
            scheduler = torch.optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lambda epoch: 1.0)
        elif args.lradj == 'WARM_COS':
            # Linear warmup then cosine decay to min_lr; stepped per iteration
            total_steps = train_steps * args.train_epochs
            warmup_steps = max(1, int(total_steps * args.warmup_ratio))
            min_lr_ratio = args.min_lr / args.learning_rate

            def warm_cos_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = min(progress, 1.0)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=warm_cos_lambda)
        elif args.lradj == 'PLATEAU_RESTART':
            # Cosine warm restarts; optional warmup handled in loop; plateau triggers restart + peak decay
            T_0_steps = max(1, int(train_steps * args.restart_T0_epochs))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                model_optim, T_0=T_0_steps, T_mult=2, eta_min=args.min_lr
            )
        elif args.lradj == 'TST':
            # OneCycleLR for TST
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)
        else:
            # OneCycleLR for other cases (type1, type2, PEMS, etc.)
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        # Loss and metric functions
        criterion = get_loss_function(args.loss)
        metric_func = get_metric_function(args.metric)

        # Prepare everything for distributed/mixed precision training
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        # AMP scaler for mixed precision
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        clip_norm = args.gradient_clip_norm
        if args.lradj == 'PLATEAU_RESTART':
            total_steps_plateau = train_steps * args.train_epochs
            warmup_steps_plateau = max(1, int(total_steps_plateau * args.warmup_ratio))
            plateau_best_vali = np.inf
            plateau_counter = 0
            current_peak_lr = args.learning_rate
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()
            
            for i, (input_data, target_data) in tqdm(enumerate(train_loader), disable=not accelerator.is_local_main_process):
                iter_count += 1
                model_optim.zero_grad()

                # Move data to accelerator device
                input_data = input_data.float().to(accelerator.device)  # [batch, seq_len, num_features]
                target_data = target_data.float().to(accelerator.device)  # [batch, pred_len, num_features]

                # Forward pass
                outputs = model(input_data)  # [batch, pred_len, num_features]
                
                # Debug: Check for NaN in inputs/outputs
                if torch.isnan(input_data).any():
                    accelerator.print(f"WARNING: NaN in input_data at iteration {i}")
                if torch.isnan(outputs).any():
                    accelerator.print(f"WARNING: NaN in outputs at iteration {i}")
                    accelerator.print(f"Output stats: min={outputs.min():.6f}, max={outputs.max():.6f}, mean={outputs.mean():.6f}")

                # Loss calculation (match validation logic)
                f_dim = -1 if args.features == 'MS' else 0
                # Extract last input price for trading loss calculation
                last_input_price = input_data[:, -1:, f_dim:] if args.features == 'MS' else input_data[:, -1:, :]
                # Pass input_data to loss function (TradingLoss uses it for risk-adjusted returns)
                try:
                    loss = criterion(outputs[:, -args.pred_len:, f_dim:], target_data[:, :, f_dim:], last_input_price)
                except TypeError:
                    # Loss function doesn't accept input_data parameter, call without it
                    loss = criterion(outputs[:, -args.pred_len:, f_dim:], target_data[:, :, f_dim:])
                
                # Debug: Check for NaN in loss
                if torch.isnan(loss):
                    accelerator.print(f"WARNING: NaN loss at iteration {i}")
                    accelerator.print(f"Pred range: {outputs.min():.6f} to {outputs.max():.6f}")
                    accelerator.print(f"Target range: {target_data.min():.6f} to {target_data.max():.6f}")
                train_loss.append(loss.item())

                # Backward pass and optimizer step
                if args.use_amp:
                    scaler.scale(loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(model_optim)
                        accelerator.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    if clip_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_norm)
                    model_optim.step()

                # Step scheduler if using per-iteration strategies (TST, WARM_COS, PLATEAU_RESTART) or OneCycleLR
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                elif args.lradj == 'WARM_COS':
                    scheduler.step()
                elif args.lradj == 'PLATEAU_RESTART':
                    global_step = epoch * train_steps + i
                    if global_step < warmup_steps_plateau:
                        lr = current_peak_lr * (global_step + 1) / warmup_steps_plateau
                        for pg in model_optim.param_groups:
                            pg['lr'] = lr
                    else:
                        scheduler.step()
                elif args.lradj not in ['COS', 'constant', 'type3']:
                    # Step OneCycleLR for other strategies that use it (type1, type2, PEMS, etc.)
                    scheduler.step()

                # Print progress every 1000 iterations
                if (i + 1) % 1000 == 0:
                    accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")

            # Calculate average training loss
            train_loss = np.average(train_loss)
            # Validation and test evaluation
            vali_loss, vali_metric = vali(args, accelerator, model, vali_data, vali_loader, criterion, metric_func)  # vali_loss: float, vali_metric: float
            test_loss, test_metric = vali(args, accelerator, model, test_data, test_loader, criterion, metric_func)  # test_loss: float, test_metric: float
            accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            accelerator.print(f"{args.metric} Metric: {test_metric:.7f}")

            # Log metrics to MLflow if enabled and on main process
            if enable_mlflow and accelerator.is_main_process:
                metrics_to_log = {
                    f"train_{args.loss.lower()}_loss": train_loss,
                    f"vali_{args.loss.lower()}_loss": vali_loss,
                    f"vali_{args.metric.lower()}_metric": vali_metric,
                    f"test_{args.loss.lower()}_loss": test_loss,
                    f"test_{args.metric.lower()}_metric": test_metric
                }
                mlflow.log_metrics(metrics_to_log, step=epoch)

            # Early stopping check
            early_stopping(vali_loss, model, temp_checkpoint_path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            # Plateau detection and restart (PLATEAU_RESTART)
            if args.lradj == 'PLATEAU_RESTART':
                if vali_loss < plateau_best_vali - args.plateau_delta:
                    plateau_best_vali = vali_loss
                    plateau_counter = 0
                else:
                    plateau_counter += 1
                if plateau_counter >= args.plateau_patience:
                    current_peak_lr = current_peak_lr * args.peak_decay
                    for pg in model_optim.param_groups:
                        pg['lr'] = current_peak_lr
                    T_0_steps = max(1, int(train_steps * args.restart_T0_epochs))
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        model_optim, T_0=T_0_steps, T_mult=2, eta_min=args.min_lr
                    )
                    plateau_best_vali = vali_loss
                    plateau_counter = 0
                    accelerator.print(f"Plateau restart: new peak LR = {current_peak_lr:.2e}")

            # Learning rate adjustment (epoch-based; skip for TST, WARM_COS, PLATEAU_RESTART which step per iteration)
            if args.lradj not in ('TST', 'WARM_COS', 'PLATEAU_RESTART'):
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                if args.lradj == 'PLATEAU_RESTART':
                    global_step_epoch = (epoch + 1) * train_steps
                    if global_step_epoch <= warmup_steps_plateau:
                        accelerator.print(f'LR (warmup) = {model_optim.param_groups[0]["lr"]:.2e}')
                    else:
                        accelerator.print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')
                else:
                    accelerator.print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        # After training, log model to MLflow if enabled and on main process
        accelerator.wait_for_everyone()
        if enable_mlflow and accelerator.is_main_process:
            # Unwrap the model to save the raw state_dict
            unwrapped_model = accelerator.unwrap_model(model)
            # Filter out frozen parameters
            state_dict = {
                k: v for k, v in unwrapped_model.state_dict().items()
                if unwrapped_model.get_parameter(k).requires_grad
            }
            mlflow.pytorch.log_state_dict(state_dict, artifact_path=None)
            accelerator.print(f"Model '{args.model_id}' has been logged to MLflow.")
            
            # Clean up temporary early stopping checkpoints
            if os.path.exists(temp_checkpoint_path):
                shutil.rmtree(temp_checkpoint_path)

def main():
    """
    Main function to set up experiment, accelerator, and start training.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Auto-detect feature set from data_path if --data not explicitly provided
    if args.data is None:
        args.data = detect_feature_set_from_path(args.data_path)
        print(f"Auto-detected prompt file from data_path: {args.data}")

    # Set random seeds for reproducibility
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Initialize Accelerator for distributed/mixed precision training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./config/ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    # Start the training process
    run_training(args, accelerator)

if __name__ == "__main__":
    main()
