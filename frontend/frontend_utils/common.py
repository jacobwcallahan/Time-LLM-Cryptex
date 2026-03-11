"""Shared configuration and helpers used across frontend tabs."""

import os
from pathlib import Path
from datetime import datetime
import logging

# Ensure project root is on path for hpo_core imports
_utils_dir = Path(__file__).resolve().parent
_frontend_dir = _utils_dir.parent
_project_root = _frontend_dir.parent
if str(_project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# MLflow configuration - override via env vars when deploying (MLFLOW_TRACKING_URI, etc.)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.1.103:5000")
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://192.168.1.103:9000"))


def _to_date_str(val):
    """Convert Gradio DateTime (datetime, timestamp, or str) to YYYY-MM-DD."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d")
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str) and len(val) >= 10:
        return val[:10]
    return str(val) if val else None
