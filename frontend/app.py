"""
Gradio frontend for Time-LLM-Cryptex HPO configuration.
Provides a UI for configuring hyperparameter optimization runs.
"""

import gradio as gr

from training import build_training_tab
from inference import build_inference_tab
from backtest import build_backtest_tab


with gr.Blocks(title="Time-LLM-Cryptex", theme=gr.themes.Citrus()) as app:
    gr.Markdown("# Time-LLM-Cryptex")
    gr.Markdown("Train, run inference, and backtest Time-LLM models for cryptocurrency forecasting.")

    with gr.Tabs():
        build_training_tab()
        build_inference_tab()
        build_backtest_tab()


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=9001, share=False, debug=True)
