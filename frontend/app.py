"""
Gradio frontend for Time-LLM-Cryptex HPO configuration.
Provides a UI for configuring hyperparameter optimization runs.
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST so utils.tools (used by models) resolves to project utils, not frontend/utils
_frontend_dir = Path(__file__).resolve().parent
_project_root = _frontend_dir.parent
_pr_str = str(_project_root)
if _pr_str in sys.path:
    sys.path.remove(_pr_str)
sys.path.insert(0, _pr_str)

# CRITICAL: frontend/utils must not exist - it shadows project utils and causes circular imports
_stale_utils = _frontend_dir / "utils"
if _stale_utils.is_dir():
    sys.exit(
        f"ERROR: Please delete the folder 'frontend/utils' and use 'frontend_utils' instead:\n"
        f"  rm -rf {_stale_utils}\n"
        f"The old 'utils' folder shadows the project's utils package (utils.tools) and causes circular imports."
    )
# Disable Gradio analytics to avoid pandas compatibility issue (infer_objects copy param)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

# Fix pandas<2.0 compatibility: Gradio calls infer_objects(copy=False) but pandas 1.x lacks 'copy' param
import pandas as _pd
_orig_df_infer = _pd.DataFrame.infer_objects
_orig_series_infer = _pd.Series.infer_objects
def _infer_objects_compat(self, copy=True):
    return _orig_df_infer(self) if isinstance(self, _pd.DataFrame) else _orig_series_infer(self)
_pd.DataFrame.infer_objects = _infer_objects_compat
_pd.Series.infer_objects = _infer_objects_compat

import gradio as gr

from training import build_training_tab
from inference import build_inference_tab
from backtest import build_backtest_tab
from experiment_runs import build_experiment_runs_tab
from custom_inference import build_custom_inference_tab


with gr.Blocks() as app:
    gr.Markdown("# Time-LLM-Cryptex")
    gr.Markdown("Train, run inference, and backtest Time-LLM models for cryptocurrency forecasting.")

    with gr.Tabs():
        build_training_tab()
        build_inference_tab()
        build_backtest_tab()
        build_experiment_runs_tab()
        build_custom_inference_tab()


if __name__ == "__main__":
    # Server: 0.0.0.0 = accept connections from any host (good for hosting)
    # share=True: get a temporary public URL via Gradio (for quick demos)
    # root_path: set when behind reverse proxy (e.g. /app for https://yoursite.com/app)
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "9001"))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("true", "1", "yes")
    root_path = os.environ.get("GRADIO_ROOT_PATH", "").strip() or None
    debug = os.environ.get("GRADIO_DEBUG", "true").lower() in ("true", "1", "yes")

    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        root_path=root_path,
        debug=debug,
        title="Time-LLM-Cryptex",
        theme=gr.themes.Citrus(),
    )
