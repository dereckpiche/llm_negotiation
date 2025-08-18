import logging
import time
from contextlib import contextmanager

import torch


def vram_usage():
    output = ""
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(i) / (
            1024**3
        )  # Convert bytes to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(i) / (
            1024**3
        )  # Convert bytes to GB
        output += f"GPU {i}: Memory Allocated: {gpu_memory_allocated:.2f} GB, Memory Reserved: {gpu_memory_reserved:.2f} GB"
    return output


def ram_usage():
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    ram_used = memory_info.rss / (1024**3)  # Convert bytes to GB
    return f"RAM Usage: {ram_used:.2f} GB"


@contextmanager
def resource_logger_context(logger: logging.Logger, task_description: str):
    """
    Context manager to log the resource usage of the current task.
    Args:
        logger: The logger to use to log the resource usage.
        task_description: The description of the task to log.
    Returns:
        None
    """
    try:
        initial_time = time.time()
        initial_gpu_allocated = torch.cuda.memory_allocated(0)
        initial_gpu_reserved = torch.cuda.memory_reserved(0)
        yield None
    finally:
        final_time = time.time()
        final_gpu_allocated = torch.cuda.memory_allocated(0)
        final_gpu_reserved = torch.cuda.memory_reserved(0)
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        logger.info(
            f"""For task: {task_description}\n,
            ΔVRAM Allocated: {(final_gpu_allocated - initial_gpu_allocated)/ (1024 ** 3)} GB,
            ΔVRAM Reserved: {(final_gpu_reserved - initial_gpu_reserved)/ (1024 ** 3)} GB,
            ΔTime: {time.strftime('%H:%M:%S', time.gmtime(final_time - initial_time))},
            Percentage of VRAM taken: {100*(final_gpu_allocated+final_gpu_reserved)/total_gpu_memory}%,
            """
        )
