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
from frontend_utils.gpu_utils import get_gpu_list_and_status


def _status_display_html(status_html: str, status: str) -> str:
    """Wrap status with label. When busy, include data attr for CSS to disable tabs."""
    busy_attr = ' data-gpu-busy="true"' if status == "busy" else ""
    hint = (
        '<span style="font-size:0.7rem;opacity:0.9;color:#94a3b8;font-style:italic;margin-left:8px;">'
        "— select an available GPU to continue</span>"
    ) if status == "busy" else ""
    return f'<div class="gpu-status-inner"{busy_attr} style="display:flex;align-items:center;gap:4px;font-size:0.8rem;"><span style="font-weight:500;">Status:</span> {status_html}{hint}</div>'


def _update_gpu_status():
    """Return status HTML for the 30s timer poll."""
    _, status, status_html = get_gpu_list_and_status()
    return _status_display_html(status_html, status)


with gr.Blocks(title="Time-LLM-Cryptex", css="""
.gradio-container { max-width: 100vw !important; width: 100vw !important; }
.app, .main, .contain { max-width: none !important; width: 100% !important; }
.gpu-bar .gr-dropdown, .gpu-bar .gr-form, .gpu-bar .gr-radio { font-size: 0.8rem !important; }
.gpu-bar .gr-form { min-height: unset !important; padding: 4px 0 !important; }
.gpu-bar { min-height: unset !important; padding: 4px 0 8px !important; width: 100% !important; min-width: 100% !important; overflow: visible !important; }
.gpu-bar > * { flex-shrink: 0 !important; overflow: visible !important; }
/* GPU form container: 245px width (overrides min(245px, 100%)), 8px left padding */
.gpu-bar .form, .gpu-bar div.form { min-width: 245px !important; width: 245px !important; padding-left: 10px !important; }
/* GPU radio fieldset: 245px width, remove grey box styling */
.gpu-bar .gpu-radio, .gpu-bar fieldset.gpu-radio, .gpu-bar .gpu-radio.block {
  min-width: 245px !important; width: 245px !important;
  overflow: visible !important; background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; border-width: 0 !important;
}
.gpu-bar .gpu-radio.padded, .gpu-bar fieldset.gpu-radio.padded { padding: 0 !important; }
.gpu-bar form, .gpu-bar .gr-form { background: transparent !important; border: none !important; }
.gpu-radio .wrap { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 8px !important; overflow: visible !important; min-width: 220px !important; }
.gpu-radio .wrap > * { font-size: 0.7rem !important; padding: 2px 8px !important; min-height: unset !important; flex-shrink: 0 !important; }
/* Shift status right to align with GPU block */
.gpu-status, .gpu-status.prose, .html-container .gpu-status { margin-left: 16px !important; }
/* Force full width for gpu-bar blocks */
.gpu-bar .block { max-width: none !important; }
/* When GPU is busy: disable all tabs, keep gpu-bar interactive */
.gradio-container:has([data-gpu-busy="true"]) .main-tabs { pointer-events: none !important; opacity: 0.7; }
.gradio-container:has([data-gpu-busy="true"]) .gpu-bar { pointer-events: auto !important; }
""") as app:
    gr.Markdown("# Time-LLM-Cryptex")
    gr.Markdown("Train, run inference, and backtest Time-LLM models for cryptocurrency forecasting.")

    with gr.Row(elem_classes=["gpu-bar"]):
        gpu_dropdown = gr.Radio(
            label="GPU",
            choices=["1", "2", "3", "4"],
            value="1",
            scale=0,
            elem_classes=["gpu-radio"],
        )
        gpu_status = gr.HTML(
            value=_status_display_html('<span style="color:#94a3b8">●</span> unknown', "unknown"),
            scale=0,
            elem_classes=["gpu-status"],
        )
        gpu_timer = gr.Timer(30, render=False)

    def _on_load():
        _, status, status_html = get_gpu_list_and_status()
        return _status_display_html(status_html, status)

    def _on_tick():
        return _update_gpu_status()

    app.load(_on_load, outputs=[gpu_status])
    gpu_timer.tick(_on_tick, outputs=[gpu_status])

    with gr.Tabs(elem_classes=["main-tabs"]):
        build_training_tab(gpu_dropdown)
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
        theme=gr.themes.Citrus(),
    )
