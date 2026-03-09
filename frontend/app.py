"""
Gradio frontend for Time-LLM-Cryptex HPO configuration.
Provides a UI for configuring hyperparameter optimization runs.
"""

import os
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


with gr.Blocks(title="Time-LLM-Cryptex", theme=gr.themes.Citrus()) as app:
    gr.Markdown("# Time-LLM-Cryptex")
    gr.Markdown("Train, run inference, and backtest Time-LLM models for cryptocurrency forecasting.")

    with gr.Tabs():
        build_training_tab()
        build_inference_tab()
        build_backtest_tab()
        build_experiment_runs_tab()
        build_custom_inference_tab()


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=9001, share=False, debug=True)
