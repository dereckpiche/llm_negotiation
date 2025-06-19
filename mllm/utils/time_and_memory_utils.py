import logging
import torch





# class Timer:
#     def __init__:
#         current_time = 

def vram_usage():
    output = ""
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert bytes to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # Convert bytes to GB
        output += f"GPU {i}: Memory Allocated: {gpu_memory_allocated:.2f} GB, Memory Reserved: {gpu_memory_reserved:.2f} GB"
    return output

def ram_usage():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    ram_used = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    return f"RAM Usage: {ram_used:.2f} GB"

