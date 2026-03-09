"""
Backtesting tab for Time-LLM-Cryptex.
Provides UI for evaluating model predictions with trading strategies.
"""

import gradio as gr

from helper_fcns import run_backtest


def build_backtest_tab():
    """Build and return the Backtesting tab UI."""
    with gr.TabItem("Backtesting", id="backtesting"):
        gr.Markdown("## Backtesting")
        gr.Markdown("Evaluate model predictions with trading strategies. Inference data is pulled from MLflow (ohlcv_inference.csv artifact).")

        with gr.Row():
            with gr.Column():
                bt_experiment_name = gr.Textbox(
                    label="Experiment Name",
                    placeholder="e.g. my_experiment",
                    info="MLflow experiment name"
                )
                bt_run_id = gr.Textbox(
                    label="Run ID",
                    placeholder="e.g. trial_abc123 or run UUID",
                    info="MLflow run ID or run name"
                )
                bt_strategy = gr.Dropdown(
                    label="Trading Strategy",
                    choices=["SimpleAI", "SLTP", "MomentumAI", "RSIAI", "BollingerAI", "MeanReversionAI", "TrendFollowingAI"],
                    value="SimpleAI",
                    info="Select backtesting strategy"
                )
                bt_initial_capital = gr.Number(
                    label="Initial Capital",
                    value=10000,
                    info="Starting capital for backtest"
                )

            with gr.Column():
                bt_start_date = gr.DateTime(
                    label="Start Date",
                    value="",
                    info="Backtest start date",
                    include_time=False
                )
                bt_end_date = gr.DateTime(
                    label="End Date",
                    value="",
                    info="Backtest end date",
                    include_time=False
                )
                bt_threshold = gr.Number(
                    label="Threshold",
                    value=0.0,
                    info="Prediction threshold for trading signals"
                )

        run_backtest_btn = gr.Button("Run Backtest", variant="primary")

        gr.Markdown("### Backtest Chart (Buys & Sells)")
        bt_equity_plot = gr.Plot(label="Price Chart with Buy/Sell Signals")

        gr.Markdown("### Backtest Summary")
        backtest_output = gr.Textbox(label="Status / Summary", lines=15, interactive=False)

        gr.Markdown("### Performance Metrics")
        with gr.Row():
            bt_total_return = gr.Number(label="Total Return (%)", interactive=False)
            bt_sharpe_ratio = gr.Number(label="Sharpe Ratio", interactive=False)
            bt_max_drawdown = gr.Number(label="Max Drawdown (%)", interactive=False)
        with gr.Row():
            bt_win_rate = gr.Number(label="Win Rate (%)", interactive=False)
            bt_num_trades = gr.Number(label="Number of Trades", interactive=False)
            bt_profit_factor = gr.Number(label="Profit Factor", interactive=False)

        run_backtest_btn.click(
            fn=run_backtest,
            inputs=[bt_experiment_name, bt_run_id, bt_strategy, bt_initial_capital, bt_start_date, bt_end_date, bt_threshold],
            outputs=[bt_equity_plot, backtest_output, bt_total_return, bt_sharpe_ratio, bt_max_drawdown, bt_win_rate, bt_num_trades, bt_profit_factor],
        )
