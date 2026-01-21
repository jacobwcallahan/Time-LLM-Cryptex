import gradio as gr
from datetime import timedelta
from datetime import datetime

def check_inf_after_train(end_date, inf_start):
    """Checks if the inference start is after the training end date.
    
    Must be used in the change event of the inference start."""
    print(f"End date: {end_date}, Inf start: {inf_start}")
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

    if start_date >= end_date:
        return gr.update(
            value=datetime.fromtimestamp(end_date - 86400.0).strftime("%Y-%m-%d"),
            info="Start date MUST be before end date."
        )
    return gr.update(info=None)

def end_after_start(end_date, start_date):
    """Checks if the end date is after the start date.
    
    Must be used in the change event of the end date."""
    if end_date is None or start_date is None:
        return gr.update()

    if end_date <= start_date:
        return gr.update(
            value=datetime.fromtimestamp(start_date + 86400.0).strftime("%Y-%m-%d"),
            info="End date MUST be after start date."
        )