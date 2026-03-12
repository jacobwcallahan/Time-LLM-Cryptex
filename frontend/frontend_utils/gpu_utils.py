"""
GPU utilities using nvidia-smi CLI.
Provides GPU list and busy/available status based on memory usage.
"""

import subprocess
import re

# A GPU is considered "full" (busy) if memory used >= this fraction of total
FULL_MEMORY_THRESHOLD = 0.15


def _run_nvidia_smi_query(query: str) -> str | None:
    """Run nvidia-smi with the given query. Returns output or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_gpu_list_and_status() -> tuple[list[tuple[str, str]], str, str]:
    """
    Query nvidia-smi for GPU list and memory usage.

    Returns:
        gpu_choices: List of (value, label) for dropdown, e.g. [("0", "GPU 0: NVIDIA A100"), ...]
        status: "busy" or "available"
        status_html: HTML string with colored dot and status text for display
    """
    out = _run_nvidia_smi_query("index,name,memory.used,memory.total")
    if not out:
        return [("0", "GPU 0 (nvidia-smi unavailable)")], "unknown", _status_html("unknown")

    gpu_choices = []
    full_count = 0
    total_count = 0

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "0, NVIDIA A100-SXM4-40GB, 1024, 40960"
        parts = [p.strip() for p in line.split(",", 3)]
        if len(parts) < 4:
            continue
        idx, name, mem_used_str, mem_total_str = parts
        try:
            mem_used = float(re.sub(r"[^\d.]", "", mem_used_str) or "0")
            mem_total = float(re.sub(r"[^\d.]", "", mem_total_str) or "1")
        except ValueError:
            mem_used, mem_total = 0.0, 1.0
        if mem_total > 0 and mem_used / mem_total >= FULL_MEMORY_THRESHOLD:
            full_count += 1
        total_count += 1
        gpu_choices.append((idx, f"GPU {idx}: {name}"))

    if total_count == 0:
        return [("0", "GPU 0")], "available", _status_html("available")

    # Busy if ANY GPU has memory usage >= threshold (all 4 must be free for available)
    status = "busy" if full_count > 0 else "available"
    return gpu_choices, status, _status_html(status)


def _status_html(status: str) -> str:
    """Return HTML for status with colored dot."""
    if status == "busy":
        return '<span style="color:#ef4444">●</span> busy'
    if status == "available":
        return '<span style="color:#22c55e">●</span> available'
    return '<span style="color:#94a3b8">●</span> unknown'
