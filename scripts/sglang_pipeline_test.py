try:
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    import subprocess, json, os, sys, time, requests
    from sglang.test.test_utils import is_in_ci

    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd
    from sglang.utils import wait_for_server, print_highlight, terminate_process

    # ----------------------------------------------------------------------
    # Helper: query GPU-memory usage with nvidia-smi (works even without pynvml)
    # ----------------------------------------------------------------------
    def gpu_mem():
        """Return used / total MiB for the first visible GPU as a string."""
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"], capture_output=True, text=True
        )
        used, total = res.stdout.strip().split(",")
        return f"{int(used):>5d} / {int(total):<5d} MiB"

    # ----------------------------------------------------------------------
    # 1️⃣  Start the server with memory saver ON
    # ----------------------------------------------------------------------
    server_process, port = launch_server_cmd(
        "python3 -m sglang.launch_server "
        "--model-path qwen/qwen2.5-0.5b-instruct "
        "--host 0.0.0.0 "
        "--enable-memory-saver"
    )
    wait_for_server(f"http://localhost:{port}")

    print(f"[BOOT]           GPU memory: {gpu_mem()}")

    # Endpoints
    gen_url     = f"http://localhost:{port}/generate"
    release_url = f"http://localhost:{port}/release_memory_occupation"
    resume_url  = f"http://localhost:{port}/resume_memory_occupation"

    # ----------------------------------------------------------------------
    # 2️⃣  First generation (warm-up)
    # ----------------------------------------------------------------------
    payload = {"text": "What is the capital of France?"}
    print_highlight(requests.post(gen_url, json=payload).json())
    print(f"[AFTER GEN #1]   GPU memory: {gpu_mem()}")

    # ----------------------------------------------------------------------
    # 3️⃣  Release weights + KV-cache
    # ----------------------------------------------------------------------
    requests.post(release_url, json={"tags": ["kv_cache"]}).raise_for_status()
    # Give CUDA a moment to free the arena
    time.sleep(2)
    print(f"[AFTER RELEASE]  GPU memory: {gpu_mem()}")

    # ----------------------------------------------------------------------
    # 4️⃣  Resume + generate again
    # ----------------------------------------------------------------------
    requests.post(resume_url, json={"tags": ["kv_cache"]}).raise_for_status()
    print_highlight(requests.post(gen_url, json=payload).json())
    print(f"[AFTER RESUME]   GPU memory: {gpu_mem()}")

    terminate_process(server_process)

except Exception:
    import traceback, sys
    traceback.print_exc()
    terminate_process(server_process)
    sys.exit(1)
