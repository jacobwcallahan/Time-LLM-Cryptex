import argparse
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
    parser.add_argument('--model_id', type=str, required=True, default='test', help='Unique identifier for the model/run')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducibility')

    # Data loader arguments
    parser.add_argument('--data', type=str, required=True, default='CRYPTEX', help='Dataset type (should be CRYPTEX)')
    parser.add_argument('--root_path', type=str, default='./dataset', help='Root path of the data files')
    parser.add_argument('--data_path', type=str, default='candlesticks-D.csv', help='Data file name')
    parser.add_argument('--features', type=str, default='MS', help='Forecasting task type: M (multivariate), S (single), MS (multi-step)')
    parser.add_argument('--target', type=str, default='close', help='Target feature for S or MS task')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Directory to save model checkpoints')

    # Forecasting task arguments
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')

    parser.add_argument('--pred_len', type=int, default=96, help='Prediction sequence length')

    # Model architecture arguments
    parser.add_argument('--enc_in', type=int, default=7, help='Encoder input size (number of features)')
    parser.add_argument('--dec_in', type=int, default=7, help='Decoder input size (number of features)')
    parser.add_argument('--c_out', type=int, default=7, help='Output size (number of features)')
    parser.add_argument('--d_model', type=int, default=16, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    parser.add_argument('--output_attention', action='store_true', help='Whether to output attention weights from encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length for patch-based models')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patch-based models')
    parser.add_argument('--prompt_domain', type=int, default=0, help='Prompt domain (for prompt-based models)')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model name (for logging/experiment tracking)')

    # Optimization and training arguments
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loader workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='Number of alignment epochs (if used)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--des', type=str, default='test', help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function for training (e.g., MSE)')
    parser.add_argument('--metric', type=str, default='MAE', help='Metric for evaluation (e.g., MAE)')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
    parser.add_argument('--pct_start', type=float, default=0.2, help='Percentage of cycle for OneCycleLR')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training (AMP)')
    parser.add_argument('--llm_layers', type=int, default=6, help='Number of LLM layers (if applicable)')
    parser.add_argument('--percent', type=int, default=100, help='Percentage of data to use')
    parser.add_argument('--num_tokens', type=int, default=1000, help='Number of tokens for mapping layer')
    parser.add_argument('--enable_mlflow', action='store_true', default=True, help='Enable MLflow logging (default: True)')

    return parser.parse_args()

def run_training(args, accelerator):
    """
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
        else:
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

                # Loss calculation (match validation logic)
                f_dim = -1 if args.features == 'MS' else 0
                loss = criterion(outputs[:, -args.pred_len:, f_dim:], target_data[:, :, f_dim:])
                train_loss.append(loss.item())

                # Backward pass and optimizer step
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                # Step scheduler if using TST strategy
                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

                # Print progress every 100 iterations
                if (i + 1) % 100 == 0:
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

            # Learning rate adjustment (if not TST)
            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                accelerator.print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        # After training, log model and scaler to MLflow if enabled and on main process
        accelerator.wait_for_everyone()
        if enable_mlflow and accelerator.is_main_process:
            # Unwrap the model to save the raw state_dict
            unwrapped_model = accelerator.unwrap_model(model)
            # Filter out frozen parameters
            state_dict = {
                k: v for k, v in unwrapped_model.state_dict().items()
                if unwrapped_model.get_parameter(k).requires_grad
            }
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model_state_dict")
            accelerator.print(f"Model '{args.model_id}' has been logged to MLflow.")
            # Save scaler for inference
            import pickle
            import tempfile
            scaler = train_data.scaler if hasattr(train_data, 'scaler') else None  # StandardScaler or None
            if scaler is not None:
                with tempfile.NamedTemporaryFile(mode='wb', suffix=".pkl", delete=False) as tmp:
                    pickle.dump(scaler, tmp)
                    mlflow.log_artifact(tmp.name, "scaler.pkl")
                os.remove(tmp.name)
                accelerator.print(f"Scaler has been logged to MLflow.")
            # Clean up temporary early stopping checkpoints
            if os.path.exists(temp_checkpoint_path):
                shutil.rmtree(temp_checkpoint_path)

def main():
    """
    Main function to set up experiment, accelerator, and start training.
    """
    # Parse command-line arguments
    args = parse_args()

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