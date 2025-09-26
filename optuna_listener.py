import json, os, time, subprocess, sys
from pathlib import Path
import argparse
import optuna
from optuna.trial import TrialState


parser = argparse.ArgumentParser()
parser.add_argument("--study_name", type=str, default="optuna_study")
args = parser.parse_args()
# --------- CONFIG ----------
STUDY_NAME = os.getenv("OPTUNA_STUDY", args.study_name)
STORAGE_URL = os.getenv("OPTUNA_STORAGE", "sqlite:////data-fast/nfs/mlflow/optuna_study.db")
STATE_FILE = Path(os.getenv("OPTUNA_LISTENER_STATE", "/data-fast/nfs/cryptex_queue/listener_state.json"))
LOG_DIR = Path(os.getenv("OPTUNA_LISTENER_LOGS", "/data-fast/nfs/cryptex_queue/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
POLL_SECS = int(os.getenv("OPTUNA_POLL_SECS", "10"))

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed": []}

def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE_FILE)


def run_inference(model_id, model, data_path):
    infer_cmd = ["python3", "run_inference.py", "--model_id", model_id, "--llm_model", model, "--data_path", data_path]

    log_file = LOG_DIR / f"{model_id}.log"
    with open(log_file, "a", buffering=1) as lf:
        lf.write(f"\n=== {time.ctime()} | Inference start: {model_id}\n")
        p1 = subprocess.Popen(infer_cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        if p1.wait() != 0:
            raise RuntimeError(f"Inference failed: {model_id}")

        if p1.wait() != 0:
            raise RuntimeError(f"Backtest failed: {model_id}")
