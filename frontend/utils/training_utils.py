"""Training tab helpers - date validation and Gradio updates."""

from datetime import datetime
import gradio as gr


def check_inf_after_train(end_date, inf_start):
    """Checks if the inference start is after the training end date.
    Must be used in the change event of the inference start."""
    if end_date is None or inf_start is None:
        return gr.update()
    if inf_start <= end_date:
        return gr.update(
            value=datetime.fromtimestamp(end_date + 86400.0).strftime("%Y-%m-%d"),
            info="Inference start MUST be after training end date."
        )
    return gr.update(info=None)


def start_before_end(start_date, end_date):
    """Checks if the start date is before the end date.
    Must be used in the change event of the start date."""
    if start_date is None or end_date is None:
        return gr.update()

    def _to_ts(val):
        if isinstance(val, (int, float)):
            return val
        if hasattr(val, "timestamp"):
            return val.timestamp()
        return None

    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    if start_ts is None or end_ts is None:
        return gr.update()
    if start_ts >= end_ts:
        return gr.update(
            value=datetime.fromtimestamp(end_ts - 86400.0).strftime("%Y-%m-%d"),
            info="Start date MUST be before end date."
        )
    return gr.update(info=None)


def end_after_start(end_date, start_date):
    """Checks if the end date is after the start date.
    Must be used in the change event of the end date."""
    if end_date is None or start_date is None:
        return gr.update()

    def _to_ts(val):
        if isinstance(val, (int, float)):
            return val
        if hasattr(val, "timestamp"):
            return val.timestamp()
        return None

    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    if start_ts is None or end_ts is None:
        return gr.update()
    if end_ts <= start_ts:
        return gr.update(
            value=datetime.fromtimestamp(start_ts + 86400.0).strftime("%Y-%m-%d"),
            info="End date MUST be after start date."
        )
    return gr.update(info=None)
