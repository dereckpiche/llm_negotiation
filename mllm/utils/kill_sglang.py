import psutil
import signal

target_name = "sglang::scheduler"
killed = []

def kill_sglang():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Some processes may not have a name or cmdline
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            if target_name in cmdline:
                print(f"Killing PID {proc.pid}: {cmdline}")
                proc.send_signal(signal.SIGKILL)
                killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
