import torch
import mlflow
import mlflow.pytorch
import pickle
from types import SimpleNamespace

def download_mlflow_artifact(run_id, artifact_path):
    """Download an artifact from MLflow and return the local path."""
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)


def load_model_and_scaler_from_mlflow(run_id, model_class):
    """Load model and scaler from MLflow run, using the provided model_class (e.g., TimeLLM.Model)."""
    # Download model state dict
    local_path = download_mlflow_artifact(run_id, "model_state_dict")
    state_dict = torch.load(local_path)

    # Get run parameters to reconstruct model
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    # Convert parameters to correct types
    args = SimpleNamespace()
    for key, value in params.items():
        if key in ['seq_len', 'pred_len', 'd_model', 'llm_layers', 
                   'num_tokens', 'patch_len', 'stride', 'enc_in', 'dec_in', 'c_out',
                   'factor', 'd_ff', 'n_heads', 'd_layers', 'moving_avg']:
            setattr(args, key, int(value))
        elif key in ['dropout']:
            setattr(args, key, float(value))
        elif key in ['output_attention']:
            setattr(args, key, value.lower() == 'true')
        else:
            setattr(args, key, value)

    # Instantiate model
    model = model_class(args)
    model.load_state_dict(state_dict, strict=False)

    # Try to load scaler if it exists
    scaler = None
    try:
        scaler_path = download_mlflow_artifact(run_id, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
    return model, args, scaler 